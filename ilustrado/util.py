# coding: utf-8
""" Catch-all file for utility functions.

TO-DO:

    * structure validation, beyond feasibility check.

"""
import numpy as np
from time import sleep
from matador.compute import FullRelaxer


class FakeFullRelaxer(FullRelaxer):
    """ Fake Relaxer for testing, with same parameters as the real one
    from matador.compute.
    """
    def __init__(self, *args, **kwargs):
        self.structure = kwargs['res']

    def relax(self, output_queue=None):
        fake_number_crunch = True
        if fake_number_crunch:
            array = np.random.rand(50, 50)
            np.linalg.eig(array)
        self.structure['enthalpy_per_atom'] = -505 + np.random.rand()
        sleep(np.random.rand())
        if np.random.rand() < 0.8:
            self.structure['optimised'] = True
        else:
            self.structure['optimised'] = False
        output_queue.put(self.structure)


def strip_useless(doc, to_run=False):
    """ Strip useless information from a matador doc.

    Parameters:

        doc (dict): structure to strip information from.

    Arguments:

        to_run (bool): whether the structure needs to be rerun,
                       i.e. whether to delete data from previous run.

    Returns:

        dict: matador document stripped of useless keys

    """
    stripped_doc = dict()
    if to_run:
        keys = ['source', 'parents', 'mutations',
                'elems', 'stoichiometry',
                'lattice_abc', 'lattice_cart',
                'positions_frac', 'num_atoms', 'atom_types']
    else:
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
