# coding: utf-8
""" Catch-all file for utility functions. """


def strip_useless(doc):
    """ Strip useless information from a matador doc. """
    stripped_doc = dict()
    keys = ['source', 'user', 'parents', 'mutations',
            'positions_frac', 'num_atoms', 'atom_types', 'stoichiometry', 'elems',
            'lattice_abc', 'lattice_cart', 'cell_volume', 'space_group',
            'enthalpy', 'enthalpy_per_atom', 'total_energy', 'total_energy_per_atom',
            'pressure', 'max_force_on_atom', 'optimised',
            'date', 'total_time_hrs', 'peak_mem_MB']
    for key in keys:
        stripped_doc[key] = doc[key]
    return stripped_doc
