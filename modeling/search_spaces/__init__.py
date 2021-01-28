from __future__ import print_function, absolute_import

from .nasbench_101 import NasBench101SearchSpace


__factory = {
    'nasbench_101': NasBench101SearchSpace
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The search sapce name.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown search space:", name)
    return __factory[name](root, *args, **kwargs)
