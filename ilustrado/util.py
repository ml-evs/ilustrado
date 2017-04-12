# coding: utf-8
""" Catch-all file for utility functions.

TO-DO:

    * structure validation
"""
import numpy as np


def strip_useless(doc):
    """ Strip useless information from a matador doc. """
    stripped_doc = dict()
    keys = ['source', 'parents', 'mutations',
            'elems', 'stoichiometry',
            'lattice_abc', 'lattice_cart', 'cell_volume', 'space_group',
            'positions_frac', 'num_atoms', 'atom_types',
            'enthalpy', 'enthalpy_per_atom', 'total_energy', 'total_energy_per_atom',
            'pressure', 'max_force_on_atom', 'optimised',
            'date', 'total_time_hrs', 'peak_mem_MB']
    for key in keys:
        if key in doc:
            stripped_doc[key] = doc[key]
            if isinstance(doc[key], np.ndarray):
                stripped_doc[key] = doc[key].tolist()
    return stripped_doc
