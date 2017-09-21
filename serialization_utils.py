import pickle
import os

def serialized_path(args):
    """
    Gets the corresponding serialized file path based on passed arguments
    """

    return "{:s}_{:s}_{:d}subjects_{:d}training.data".format(
        args.method,
        args.imgdb,
        args.subjects,
        args.img_per_subject
    )

def check_query_params(args):
    """
    True if there is a saved file referring to args, else false.

    Keyword arguments:
    args --
            img_per_subject
            imgdb
            method
            subjects
            test_img_per_subject
    """

    path = serialized_path(args)

    return os.path.isfile(path)

def save_query_params(query_params, args):
    """
    Serializes and saves the calculated projections with their corresponding
    number of subject.

    Keyword arguments:
    query_params -- calculated query_params to store
    args --
            img_per_subject
            imgdb
            method
            subjects
            test_img_per_subject
    """

    path = serialized_path(args)

    with open(path, 'wb') as f:
        pickle.dump(query_params, f)

def load_query_params(args):
    """
    Loads the serialized query_params from the specified path

    Keyword arguments:
    args --
            img_per_subject
            imgdb
            method
            subjects
            test_img_per_subject
    """

    path = serialized_path(args)

    with open(path, 'rb') as f:
        query_params = pickle.load(f)

    return query_params
