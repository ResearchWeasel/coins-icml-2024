"""
Module containing helper functions for (de)serializing and (de)compressing Python objects from/to local storage.
"""

import _pickle as c_pickle

import compress_pickle


def save_object(obj, filepath: str, compress: bool = True):
    """
    Serialize and optionally compress a Python object from memory for storage into a local binary file

    :param obj: any Python object
    :param filepath: local file system path where to store the object, string
    :param compress: whether to compress the object using gzip, boolean, default value is True
    """

    with open(filepath, "wb") as file:
        if compress:
            compress_pickle.dump(obj, file, compression="gzip", set_default_extension=True)
        else:
            c_pickle.dump(obj, file)
        file.close()


def load_object(filepath: str, compressed: bool = True):
    """
    Deserialize (and decompress) a Python object from local storage and load it into memory

    :param filepath: local file system path where the binary file is stored, string
    :param compressed: whether the object was compressed before storage, boolean, default value is True

    :return: the Python object
    """

    if compressed:
        obj = compress_pickle.load(filepath)
    else:
        with open(filepath, "rb") as file:
            obj = c_pickle.load(file)
            file.close()
    return obj
