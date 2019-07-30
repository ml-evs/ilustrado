# coding: utf-8

""" Catch-all file for utility functions.

"""

import sys
import logging

import numpy as np
from matador.compute import FullRelaxer

LOG = logging.getLogger("ilustrado")
LOG.setLevel(logging.DEBUG)


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
        keys = [
            "source",
            "parents",
            "mutations",
            "elems",
            "stoichiometry",
            "lattice_abc",
            "lattice_cart",
            "positions_frac",
            "num_atoms",
            "atom_types",
        ]
    else:
        keys = [
            "source",
            "parents",
            "mutations",
            "elems",
            "stoichiometry",
            "lattice_abc",
            "lattice_cart",
            "cell_volume",
            "space_group",
            "positions_frac",
            "num_atoms",
            "atom_types",
            "enthalpy",
            "enthalpy_per_atom",
            "total_energy",
            "total_energy_per_atom",
            "pressure",
            "max_force_on_atom",
            "optimised",
            "date",
            "total_time_hrs",
            "peak_mem_MB",
        ]

    for key in keys:
        if key in doc:
            stripped_doc[key] = doc[key]
            if isinstance(doc[key], np.ndarray):
                stripped_doc[key] = doc[key].tolist()
    return stripped_doc


class FakeFullRelaxer(FullRelaxer):
    """ Fake Relaxer for testing, with same parameters as the real one
    from matador.compute.

    """

    def __init__(self, *args, **kwargs):
        self.structure = kwargs["res"]
        self.output_queue = kwargs["output_queue"]

    def relax(self):
        fake_number_crunch = True
        if fake_number_crunch:
            size = np.random.randint(low=3, high=50)
            array = np.random.rand(size, size)
            np.linalg.eig(array)
        self.structure["enthalpy_per_atom"] = -505 + np.random.rand()
        self.structure["enthalpy"] = self.structure["enthalpy_per_atom"] * self.structure["num_atoms"]
        if np.random.rand() < 0.8:
            self.structure["optimised"] = True
        else:
            self.structure["optimised"] = False
        self.output_queue.put(self.structure)


class NewbornProcess:
    """ Simple container of process data. """

    def __init__(self, newborn_id, node, process, ncores=None):
        self.newborn_id = newborn_id
        self.node = node
        self.process = process
        self.ncores = ncores


class AseRelaxation:
    """ Perform relaxation with ASE LJ or EMT. """

    def __init__(self, doc, queue, pot_type="LJ"):
        from copy import deepcopy
        from matador.utils.viz_utils import doc2ase
        from ase.calculators.lj import LennardJones
        from ase.calculators.emt import EMT

        if pot_type == "LJ":
            self.calc = LennardJones()
        else:
            self.calc = EMT()

        self.doc = deepcopy(doc)
        self.atoms = doc2ase(doc)
        self.atoms.set_calculator(self.calc)
        self.queue = queue

    def relax(self):
        from ase.optimize import LBFGS

        cached = sys.__stdout__
        # sys.stdout = os.devnull
        try:
            optimizer = LBFGS(self.atoms)
            optimizer.logfile = None
            optimised = optimizer.run(steps=50)
        except Exception:
            optimised = False

        self.doc["optimised"] = bool(optimised)
        self.doc["positions_frac"] = self.atoms.get_scaled_positions().tolist()
        self.doc["lattice_cart"] = self.atoms.cell.tolist()
        self.doc["enthalpy_per_atom"] = float(self.calc.results["energy"] / len(
            self.doc["atom_types"]
        ))
        self.doc["enthalpy"] = float(self.calc.results["energy"])
        self.queue.put(self.doc)
        sys.stdout = cached
