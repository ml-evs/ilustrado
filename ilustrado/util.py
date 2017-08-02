# coding: utf-8
""" Catch-all file for utility functions.

TO-DO:

    * structure validation, beyond feasibility check.

"""
import numpy as np
from time import sleep


class FakeFullRelaxer(object):
    """ Fake Relaxer for testing, with same parameters as the real one
    from matador.compute.
    """
    def __init__(self, res, param_dict, cell_dict,
                 ncores, nnodes, node,
                 executable='castep', rough=None, spin=False,
                 reopt=False, custom_params=False, killcheck=False,
                 kpts_1D=False, conv_cutoff=None, conv_kpt=None, archer=False, bnl=False,
                 start=True, redirect=False, verbosity=0, debug=False):
        self.structure = res

    def relax(self, output_queue=None):
        self.structure['enthalpy_per_atom'] = np.random.rand()
        sleep(np.random.rand())
        if np.random.rand() < 0.8:
            self.structure['optimised'] = True
        else:
            self.structure['optimised'] = False
        output_queue.put(self.structure)


def strip_useless(doc):
    """ Strip useless information from a matador doc.

    Input:

        | doc: dict, structure to strip information from.

    """
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
