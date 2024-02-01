"""
Module containing the implementation of the link prediction models from related work.
"""
import attr
import torch as pt
from attr.validators import and_, ge, in_, instance_of, le
from torch import nn
from torch.linalg import norm
from torch.nn.functional import relu
from torch_geometric.nn import MLP

from graph_completion.utils import AbstractConf


def distance_TransE(query_emb: pt.Tensor, answer_emb: pt.Tensor, p: int = 2) -> pt.Tensor:
    return norm(query_emb - answer_emb, ord=p, dim=-1)


def score_DistMult(query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
    return pt.sum(query_emb * answer_emb, dim=-1)


def score_ComplEx(query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
    query_real, query_imag = pt.chunk(query_emb, 2, dim=-1)
    answer_real, answer_imag = pt.chunk(answer_emb, 2, dim=-1)

    score_real = query_real * answer_real + query_imag * answer_imag
    return pt.sum(score_real, dim=-1)


def distance_RotatE(query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
    query_real, query_imag = pt.chunk(query_emb, 2, dim=-1)
    answer_real, answer_imag = pt.chunk(answer_emb, 2, dim=-1)

    diff_real = query_real - answer_real
    diff_imag = query_imag - answer_imag
    diff_abs = norm(pt.stack([diff_real, diff_imag], dim=-1), dim=-1)

    return norm(diff_abs, ord=1, dim=-1)


def score_Context(query_emb: pt.Tensor, ctx_emb: pt.Tensor) -> pt.Tensor:
    return pt.einsum("...i,...i", query_emb, ctx_emb)


def distance_Query2Box(query_emb: pt.Tensor, answer_emb: pt.Tensor, alpha: float) -> pt.Tensor:
    query_center, query_offset = pt.chunk(query_emb, 2, dim=-1)
    answer_center, answer_offset = pt.chunk(answer_emb, 2, dim=-1)
    query_min, query_max = query_center - query_offset, query_center + query_offset

    dist_outside = norm(relu(answer_center - query_max) + relu(query_min - answer_center), ord=1, dim=-1)
    dist_inside = norm(query_center - pt.minimum(query_max, pt.maximum(query_min, answer_center)), ord=1, dim=-1)

    return dist_outside + alpha * dist_inside


def distance_BetaE(query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
    query_alpha, query_beta = pt.chunk(query_emb, 2, dim=-1)
    answer_alpha, answer_beta = pt.chunk(answer_emb, 2, dim=-1)

    query_prob = pt.distributions.Beta(query_alpha, query_beta)
    answer_prob = pt.distributions.Beta(answer_alpha, answer_beta)
    return pt.sum(pt.distributions.kl_divergence(answer_prob, query_prob), dim=-1)


class LinkPredictorMLP(nn.Module):
    """
    Class representing a general MLP edge scorer.
    """

    def __init__(self, embedding_dim: int, mlp_hidden_dim: int, mlp_num_hidden_layers: int):
        super().__init__()
        self.hidden_dim = mlp_hidden_dim
        self.num_hidden_layers = mlp_num_hidden_layers

        self.mlp = MLP([2 * embedding_dim, ] + [mlp_hidden_dim, ] * mlp_num_hidden_layers + [1, ])

    def forward(self, query_emb: pt.Tensor, ctx_emb: pt.Tensor) -> pt.Tensor:
        return self.mlp(pt.column_stack((query_emb, ctx_emb))).squeeze(1)


class LinkPredictorTransE(nn.Module):
    """
    Class implementing the TransE scoring function.
    """

    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
        return self.margin - distance_TransE(query_emb, answer_emb)


class LinkPredictorDistMult(nn.Module):
    """
    Class implementing the DistMult scoring function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
        return score_DistMult(query_emb, answer_emb)


class LinkPredictorComplex(nn.Module):
    """
    Class implementing the ComplEx scoring function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
        return score_ComplEx(query_emb, answer_emb)


class LinkPredictorRotatE(nn.Module):
    """
    Class implementing the RotatE scoring function.
    """

    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
        return self.margin - distance_RotatE(query_emb, answer_emb)


class LinkPredictorGATNE(nn.Module):
    """
    Class implementing the GATNE scoring function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query_emb: pt.Tensor, ctx_emb: pt.Tensor) -> pt.Tensor:
        return score_Context(query_emb, ctx_emb)


class LinkPredictorSACN(nn.Module):
    """
    Class representing the SACN link scoring model.
    """

    def __init__(self, embedding_dim: int,
                 sacn_num_conv_channels: int, sacn_conv_kernel_size: int, sacn_dropout_rate: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_conv_channels = sacn_num_conv_channels
        self.conv_kernel_size = sacn_conv_kernel_size
        self.dropout_rate = sacn_dropout_rate

        self.inp_drop = nn.Dropout(sacn_dropout_rate)
        self.hidden_drop = nn.Dropout(sacn_dropout_rate)
        self.feature_map_drop = nn.Dropout(sacn_dropout_rate)
        self.conv1 = nn.Conv1d(2, sacn_num_conv_channels, sacn_conv_kernel_size,
                               stride=1, padding=sacn_conv_kernel_size // 2)
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(sacn_num_conv_channels)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.fc = nn.Linear(embedding_dim * sacn_num_conv_channels, embedding_dim)

    def forward(self, query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
        query_emb = pt.stack(pt.chunk(query_emb, 2, dim=-1), dim=1)
        if query_emb.size(0) > 1:
            query_emb = self.bn0(query_emb)
        x = self.inp_drop(query_emb)
        x = self.conv1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.num_conv_channels * self.embedding_dim)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = relu(x)
        y = pt.sigmoid(pt.sum(x * answer_emb, dim=1))
        return y


class LinkPredictorKBGAT(nn.Module):
    """
    Class representing the KBGAT link scoring model.
    """

    def __init__(self, embedding_dim: int, kbgat_num_conv_channels: int, kbgat_dropout_rate: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_conv_channels = kbgat_num_conv_channels
        self.dropout_rate = kbgat_dropout_rate

        self.conv_layer = nn.Conv2d(1, kbgat_num_conv_channels, (1, 3))
        self.dropout = nn.Dropout(kbgat_dropout_rate)
        self.fc_layer = nn.Linear(embedding_dim * kbgat_num_conv_channels, 1)

    def forward(self, query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
        x = pt.stack((*pt.chunk(query_emb, 2, dim=-1), answer_emb), dim=2).unsqueeze(1)
        x = self.dropout(relu(self.conv_layer(x)))
        x = x.view(-1, self.num_conv_channels * self.embedding_dim)
        y = self.fc_layer(x).squeeze(1)
        return y


class LinkPredictorQuery2Box(nn.Module):
    """
    Class implementing the Query2Box scoring function.
    """

    def __init__(self, margin: float, q2b_alpha: float):
        super().__init__()
        self.margin = margin
        self.alpha = q2b_alpha

    def forward(self, query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
        return self.margin - distance_Query2Box(query_emb, answer_emb, self.alpha)


class LinkPredictorBetaE(nn.Module):
    """
    Class implementing the BetaE scoring function.
    """

    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
        return self.margin - distance_BetaE(query_emb, answer_emb)


@attr.s
class LinkRankerHpars(AbstractConf):
    OPTIONS = {"mlp": LinkPredictorMLP, "transe": LinkPredictorTransE, "distmult": LinkPredictorDistMult,
               "complex": LinkPredictorComplex, "rotate": LinkPredictorRotatE,
               "gatne": LinkPredictorGATNE, "sacn": LinkPredictorSACN, "kbgat": LinkPredictorKBGAT,
               "gqe": LinkPredictorTransE, "q2b": LinkPredictorQuery2Box, "betae": LinkPredictorBetaE}
    algorithm = attr.ib(validator=in_(list(OPTIONS.keys())))
    embedding_dim = attr.ib(validator=instance_of(int))
    mlp_hidden_dim = attr.ib(default=128, validator=instance_of(int))
    mlp_num_hidden_layers = attr.ib(default=2, validator=instance_of(int))
    margin = attr.ib(default=1.0, validator=instance_of(float))
    sacn_num_conv_channels = attr.ib(default=100, validator=instance_of(int))
    sacn_conv_kernel_size = attr.ib(default=5, validator=instance_of(int))
    sacn_dropout_rate = attr.ib(default=0.2, validator=and_(instance_of(float), ge(0), le(1)))
    kbgat_num_conv_channels = attr.ib(default=500, validator=instance_of(int))
    kbgat_dropout_rate = attr.ib(default=0.0, validator=and_(instance_of(float), ge(0), le(1)))
    q2b_alpha = attr.ib(default=0.02, validator=instance_of(float))

    def __attrs_post_init__(self):
        self.name = self.algorithm
