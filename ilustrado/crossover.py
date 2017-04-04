# coding: utf-8
""" This file implements crossover functionality. """


def crossover(parents, method='random_slice'):
    if method is 'random_slice':
        _crossover = random_slice
    elif method is 'periodic_cut':
        _crossover = periodic_cut

    return _crossover(parents)


def random_slice(parents):
    """ Random reflection, rotation and slicing
    a la XtalOpt.
    """
    child = dict()
    raise NotImplementedError
    return child


def periodic_cut(parents):
    """ Periodic cut a la CASTEP/Abraham & Probert. """
    child = dict()
    raise NotImplementedError
    return child
