""" This file implements the GA algorithm and acts as main(). """
# standard library
import multiprocessing as mp
import subprocess as sp
import logging
import glob
import shutil
import os
import time
import sys
from traceback import print_exc
from json import dumps, dump
from copy import deepcopy, copy

# external libraries
import numpy as np
from pkg_resources import require

# matador modules
import matador.compute
import matador.compute.slurm
from matador.scrapers.castep_scrapers import (
    res2dict,
    castep2dict,
    cell2dict,
    param2dict,
)
from matador.export import doc2res
from matador.export.utils import generate_hash
from matador.fingerprints.similarity import get_uniq_cursor
from matador.fingerprints.pdf import PDFFactory
from matador.utils.chem_utils import get_formula_from_stoich, get_root_source
from matador.hull import QueryConvexHull

# ilustrado modules
from .adapt import adapt
from .generation import Generation
from .fitness import FitnessCalculator
from .util import strip_useless, LOG, NewbornProcess

__version__ = require("ilustrado")[0].version


# As this class has many settings that are hacked directly into __dict__, disable these warnings.
# pylint: disable=access-member-before-definition
# pylint: disable=attribute-defined-outside-init
# pylint: disable bad-continuation
class ArtificialSelector:
    """ ArtificialSelector takes an initial gene pool
    and applies a genetic algorithm to optimise some
    fitness function.

    Keyword Arguments:

        gene_pool (list(dict))  : initial cursor to use as "Generation 0",
        seed (str)              : seed name of cell and param files for CASTEP,
        seed_prefix (str)       : if not specifying a seed, this name will prefix all runs
        fitness_metric (str)    : currently either 'hull' or 'test',
        hull (QueryConvexHull)  : matador QueryConvexHull object to calculate distances,
        res_path (str)          : path to folder of res files to create hull, if no hull object passed
        mutation_rate (float)   : rate at which to perform single-parent mutations (DEFAULT: 0.5)
        crossover_rate (float)  : rate at which to perform crossovers (DEFAULT: 0.5)
        num_generations (int)   : number of generations to breed before quitting (DEFAULT: 5)
        num_survivors (int)     : number of structures to survive to next generation for breeding
                                  (DEFAULT: 10)
        population (int)        : number of structures to breed in any given generation
                                  (DEFAULT: 25)
        failure_ratio (int)     : maximum number of attempts per success (DEFAULT: 5)
        elitism (float)         : fraction of next generation to be comprised of elite
                                  structures from previous generation (DEFAULT: 0.2)
        best_from_stoich (bool) : whether to always include the best structure from a
                                 stoichiomtery in the next generation,
        mutations (list(str))   : list of mutation names to use,
        structure_filter (fn(doc)) : any function that takes a matador doc and returns True
                                     or False,
        check_dupes (bool)         : if True, filter relaxed structures for uniqueness on-the-fly (DEFAULT: True)
        check_dupes_hull (bool)    : compare pdf with all hull structures (DEFAULT: True)
        sandbagging (bool)         : whether or not to disfavour nearby compositions (DEFAULT: False)
        minsep_dict (dict)         : dictionary containing element-specific minimum separations, e.g.
                                     {('K', 'K'): 2.5, ('K', 'P'): 2.0}. These should only be set such that
                                     atoms do not overlap; let the DFT deal with bond lengths. No effort is made
                                     to push apart atoms that are too close, the trial will simply be discarded. (DEFAULT: None)
        max_num_mutations (int)    : maximum number of mutations to perform on a single structure,
        max_num_atoms (int)        : most atoms allowed in a structure post-mutation/crossover,
        nodes (list(str))          : list of node names to run on,
        ncores (int or list(int))  : specifies the number of cores used by listed `nodes` per thread,
        nprocs (int)               : total number of processes,
        recover_from (str)         : recover from previous run_hash, by default ilustrado will recover
                                     if it finds only one run hash in the folder
        load_only (bool)           : only load structures, do not continue breeding (DEFAULT: False)
        executable (str)           : path to DFT binary (DEFAULT: castep)
        compute_mode (str)         : either `direct`, `slurm`, `manual` (DEFAULT: direct)
        max_num_nodes (int)        : amount of array jobs to run per generation in `slurm` mode,
        walltime_hrs (int)         : maximum walltime for a SLURM array job,
        slurm_template (str)       : path to template slurm script that includes module loads etc,
        entrypoint (str)           : path to script that initialised this object, such that it can
                                     be called by SLURM
        debug (bool)               : maximum printing level
        testing (bool)             : run test code only if true
        verbosity (int)            : extra printing level,
        loglevel (str)             : follows std library logging levels.

    """

    def __init__(self, **kwargs):
        """ This is the main entrypoint. Initialises parameters,
        gene pool and begins the GA.
        """
        prop_defaults = {
            # important, required parameters
            "gene_pool": None,
            "seed": None,
            "seed_prefix": None,
            "fitness_metric": "hull",
            "hull": None,
            "res_path": None,
            # recovery and loading parameters
            "recover_from": None,
            "load_only": False,
            # GA numerical parameters
            "mutation_rate": 1.0,
            "crossover_rate": 0.0,
            "num_generations": 5,
            "num_survivors": 10,
            "population": 25,
            "elitism": 0.2,
            "max_num_mutations": 3,
            "max_num_atoms": 30,
            # other GA options
            "best_from_stoich": True,
            "mutations": None,
            "structure_filter": None,
            "check_dupes": True,
            "check_dupes_hull": True,
            "failure_ratio": 5,
            "sandbagging": False,
            "minsep_dict": None,
            # logistical and compute parameters
            "compute_mode": "direct",
            "nodes": None,
            "ncores": None,
            "nprocs": 1,
            "relaxer_params": None,
            "executable": "castep",
            "max_num_nodes": None,
            "walltime_hrs": None,
            "slurm_template": None,
            "entrypoint": None,
            # debug and logging parameters
            "debug": False,
            "testing": False,
            "emt": False,
            "verbosity": 0,
            "loglevel": "info",
        }

        # cache current params to reload again later
        self.current_params = deepcopy(prop_defaults)
        self.current_params.update(kwargs)

        self.__dict__.update(prop_defaults)
        self.__dict__.update(kwargs)

        splash_screen = (
            r"   _  _              _                     _" + "\n"
            r"  (_)| |            | |                   | |" + "\n"
            r"   _ | | _   _  ___ | |_  _ __   __ _   __| |  ___" + "\n"
            r"  | || || | | |/ __|| __|| '__| / _` | / _` | / _ \ " + "\n"
            r"  | || || |_| |\__ \| |_ | |   | (_| || (_| || (_) |" + "\n"
            r"  |_||_| \__,_||___/ \__||_|    \__,_| \__,_| \___/" + "\n\n"
            "****************************************************\n"
        )
        print("\033[92m\033[1m")
        print("\n" + splash_screen)
        print("\033[0m")

        print("Loading harsh realities of life...", end="")
        # post-load checks
        if self.relaxer_params is None:
            self.relaxer_params = dict()
        self.next_gen = None
        if isinstance(self.ncores, list):
            if len(self.ncores) != len(self.nodes):
                raise RuntimeError(
                    "Length mismatch between ncores and nodes list: {} vs {}".format(
                        self.ncores, self.nodes
                    )
                )

        # set up computing resource
        if self.compute_mode not in ("slurm", "direct", "manual"):
            raise RuntimeError("`compute_mode` must be one of `slurm`, `direct`, `manual`.")

        if self.compute_mode == "slurm":
            errors = []
            if not isinstance(self.walltime_hrs, int):
                errors.append(
                    "`walltime_hrs` specified incorrectly {}".format(self.walltime_hrs)
                )
            elif not self.walltime_hrs > 0:
                errors.append(
                    "`walltime_hrs` specified incorrectly {}".format(self.walltime_hrs)
                )
            if not isinstance(self.max_num_nodes, int):
                errors.append(
                    "`max_num_nodes` specified incorrectly {}".format(
                        self.max_num_nodes
                    )
                )
            elif not self.max_num_nodes > 0:
                errors.append(
                    "`max_num_nodes` specified incorrectly {}".format(
                        self.max_num_nodes
                    )
                )
            if not isinstance(self.slurm_template, str):
                errors.append(
                    "`slurm_template` must be a valid path, not {}".format(
                        self.slurm_template
                    )
                )
            elif not os.path.isfile(self.slurm_template):
                errors.append(
                    "`slurm_template` file {} does not exist".format(
                        self.slurm_template
                    )
                )

            if errors:
                raise RuntimeError(
                    "Invalid specification for `compute_mode='slurm'`, errors: \n{}".format(
                        "\n".join(errors)
                    )
                )

            self.slurm_dict = matador.compute.slurm.get_slurm_env()

        if self.compute_mode == "direct":
            if self.nodes is not None:
                if self.nprocs != len(self.nodes):
                    logging.warning(
                        "Specified procs {} being replaced by number of nodes {}".format(
                            self.nprocs, len(self.nodes)
                        )
                    )
                    self.nprocs = len(self.nodes)

        # set up GA logistics
        self.run_hash = generate_hash()
        self.generations = []  # list to store all generations
        self.num_elite = int(self.elitism * self.num_survivors)
        self.num_accepted = self.num_survivors - self.num_elite
        self.max_attempts = self.failure_ratio * self.population

        if self.num_survivors > self.population + self.num_elite:
            raise RuntimeError(
                "More survivors than total population: {} vs {}".format(
                    self.num_survivors, self.population + self.num_elite
                )
            )

        if self.num_accepted > self.population:
            raise RuntimeError(
                "More accepted than total population: {} vs {}".format(
                    self.num_accepted, self.population + self.num_elite
                )
            )

        if self.mutations is not None and isinstance(self.mutations, str):
            self.mutations = [self.mutations]

        if not isinstance(self.max_num_mutations, int) and self.max_num_mutations < 0:
            raise RuntimeError(
                "`max_num_mutations` must be >= 0, not {}".format(
                    self.max_num_mutations
                )
            )

        if not isinstance(self.max_num_atoms, int) and self.max_num_atoms < 1:
            raise RuntimeError(
                "`max_num_atoms` must be >= 1, not {}".format(self.max_num_atoms)
            )

        # recover from specified run
        if self.recover_from is not None:
            if isinstance(self.recover_from, str):
                self.run_hash = self.recover_from.split("/")[-1]
        # try to look for gen0 files, if multiple are found, safely exit
        else:
            gen0_files = glob.glob("*gen0.json")
            if len(gen0_files) > 1:
                raise SystemExit(
                    "Several incomplete runs found in this folder, please tidy up before re-running."
                )
            if len(gen0_files) == 1:
                self.run_hash = gen0_files[0].split("/")[-1].replace("-gen0.json", "")
                self.recover_from = self.run_hash
            else:
                print("No recovery possible, starting fresh run.")

        # set up logging
        numeric_loglevel = getattr(logging, self.loglevel.upper(), None)
        if not isinstance(numeric_loglevel, int):
            raise SystemExit(
                self.loglevel,
                "is an invalid log level, please use either `info`, `debug` or `warning`.",
            )
        file_handler = logging.FileHandler(self.run_hash + ".log", mode="a")
        file_handler.setLevel(numeric_loglevel)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s | %(levelname)8s: %(message)s")
        )
        LOG.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(numeric_loglevel)
        stream_handler.setFormatter(
            logging.Formatter("stdout %(asctime)s - %(name)s | %(levelname)8s: %(message)s")
        )
        LOG.addHandler(stream_handler)

        LOG.info("Starting up ilustrado {}".format(__version__))

        # initialise fitness calculator
        if self.fitness_metric == "hull" and self.hull is None:
            if self.res_path is not None and os.path.isfile(self.res_path):
                res_files = glob.glob("{}/*.res".format(self.res_path))
                if not res_files:
                    raise SystemExit("No structures found in {}".format(self.res_path))
                self.cursor = []
                for res in res_files:
                    self.cursor.append(res2dict(res))
                self.hull = QueryConvexHull(cursor=self.cursor)
            raise SystemExit(
                "Need to pass a QueryConvexHull object to use hull distance metric."
            )
        if self.fitness_metric in ["dummy", "hull_test"]:
            self.testing = True

        if self.testing and self.compute_mode == "slurm":
            raise SystemExit("Please use `compute_mode=direct` for testing.")

        print("Done!")
        self.fitness_calculator = FitnessCalculator(
            fitness_metric=self.fitness_metric,
            hull=self.hull,
            sandbagging=self.sandbagging,
            debug=self.debug,
        )
        LOG.debug("Successfully initialised fitness calculator.")

        # if we're checking hull pdfs too, make this list now
        if self.check_dupes_hull:
            print("Computing extra PDFs from hull...")
            PDFFactory(self.hull.cursor)
            self.extra_pdfs = deepcopy(self.hull.cursor)
            # remove pdf object from cursor so generation can be serialized
            for ind, _ in enumerate(self.hull.cursor):
                del self.hull.cursor[ind]["pdf"]
        else:
            self.extra_pdfs = None
        LOG.info("Successfully initialised similarity lists.")

        if self.recover_from is not None:
            print("Attempting to recover from run {}".format(self.run_hash))
            if isinstance(self.recover_from, str):
                LOG.info(
                    "Attempting to recover from previous run {}".format(self.run_hash)
                )
            self.recover()

        if not self.load_only:
            self.start()

    def start(self):
        """ Start running GA. """
        print("Initialising quantum mechanics...", end=" ")
        # read parameters for relaxation from seed files
        if self.seed is not None:
            seed = self.seed
            errors = []
            self.cell_dict, success_cell = cell2dict(seed, db=False)
            self.param_dict, success_param = param2dict(seed, db=False)
            if not success_cell:
                errors.append("Failed to read cell file: {}".format(self.cell_dict))
            if not success_param:
                errors.append("Failed to read param file: {}".format(self.param_dict))
            if errors:
                raise RuntimeError("{}".format(errors.join("\n")))

        else:
            self.seed = "ilustrado"
            if self.seed_prefix is not None:
                self.seed = self.seed_prefix

            self.cell_dict = {}
            self.param_dict = {}

        print("Done!\n")
        LOG.debug("Successfully initialised cell and param files.")

        if self.recover_from is None:
            self.seed_generation_0(self.gene_pool)

        if self.debug:
            print(self.nodes)
        if self.nodes is not None:
            LOG.info("Running on nodes: {}".format(" ".join(self.nodes)))
        elif self.compute_mode == "slurm":
            LOG.info("Running through SLURM queue")
        else:
            LOG.info("Running on localhost only")

        if self.debug:
            print(
                "Current number of generations: {}. Target number: {}".format(
                    len(self.generations), self.num_generations
                )
            )
        # run GA self.num_generations
        while len(self.generations) < self.num_generations:
            self.breed_generation()
            LOG.info("Successfully bred generation {}".format(len(self.generations)))

        assert len(self.generations) == self.num_generations
        self.finalise_files_for_export()
        print("Reached target number of generations!")
        print("Completed GA!")
        LOG.info("Reached target number of generations!")
        LOG.info("Completed GA!")

    def breed_generation(self):
        """ Build next generation from mutations/crossover of current and
        perform relaxations if necessary.
        """
        # initialise next_gen
        if self.next_gen is None:
            self.next_gen = Generation(
                self.run_hash,
                len(self.generations),
                self.num_survivors,
                self.num_accepted,
                fitness_calculator=self.fitness_calculator,
            )

        # newborns is a list of structures, initially raw then relaxed
        if self.compute_mode == "direct":
            self.continuous_birth()
        elif self.compute_mode in ("slurm", "manual"):
            self.batch_birth()

        if len(self.next_gen) < self.population:
            LOG.warning("Next gen is smaller than desired population.")
        # assert len(self.next_gen) >= self.population

        self.next_gen.rank()
        LOG.info("Ranked structures in generation {}".format(len(self.generations)))
        if not self.testing:
            cleaned = self.next_gen.clean()
            LOG.info(
                "Cleaned structures in generation {}, removed {}".format(
                    len(self.generations), cleaned
                )
            )

        self.enforce_elitism()
        self.reset_and_dump()
        print(self.generations[-1])

    def write_unrelaxed_generation(self):
        """ Perform mutations and write res files for the resulting
        structures. Additionally, dump an unrelaxed json file.
        """
        while len(self.next_gen) < self.max_attempts:
            newborn = self.birth_new_structure()
            self.next_gen.birth(newborn)
        for newborn in self.next_gen:
            newborn = strip_useless(newborn)
            doc2res(newborn, newborn["source"][0], info=False)
            self.next_gen.dump("unrelaxed")

    def batch_birth(self):
        """ Assess whether a generation has been relaxed already. This is done by
        checking for the existence of a file called <run_hash>-genunrelaxed.json.

        If so, match the relaxations up with the cached unrelaxed structures
        and rank them ready for the next generation.

        If not, create a new generation of structures, dump the unrelaxed structures to file,
        create the jobscripts to relax them, submit them and the job to check up on the relaxations,
        then exit.

        """

        print("Beginning birthing of generation {}...".format(len(self.generations)))
        fname = "{}-genunrelaxed.json".format(self.run_hash)
        if os.path.isfile(fname):
            LOG.info("Found existing generation to be relaxed...")
            # load the unrelaxed structures into a dummy generation
            assert os.path.isfile(fname)
            unrelaxed_gen = Generation(
                self.run_hash,
                len(self.generations),
                self.num_survivors,
                self.num_accepted,
                dumpfile=fname,
                fitness_calculator=None,
            )
            # check to see which unrelaxed structures completed successfully
            LOG.info("Scanning for completed relaxations...")
            for _, newborn in enumerate(unrelaxed_gen):
                completed_castep_filename = "completed/{}.castep".format(newborn["source"][0])
                completed_res_filename = "completed/{}.res".format(newborn["source"][0])
                doc = None
                s = None
                if os.path.isfile(completed_castep_filename):
                    doc, s = castep2dict(completed_res_filename, db=True)
                elif os.path.isfile(completed_res_filename):
                    doc, s = res2dict(completed_res_filename, db=True)
                    # if we find a res file in a completed folder, assumed it was relaxed
                    doc["optimised"] = True

                # if all was a success, then "birth" the structure, after checking for uniqueness
                if s and isinstance(doc, dict):
                    newborn = strip_useless(newborn)
                    doc = strip_useless(doc)
                    newborn.update(doc)
                    assert newborn.get("parents") is not None
                    LOG.info("Scraping result for {}".format(newborn["source"][0]))
                    self.scrape_result(newborn)
                else:
                    LOG.warning(
                        "Failed to add {}, data found: {}".format(newborn["source"][0], doc)
                    )

            # if there are not enough unrelaxed structures after that run, clean up then resubmit
            LOG.info(
                "Found {} structures out of target {}".format(
                    len(self.next_gen), self.population
                )
            )
            if len(self.next_gen) < self.population:
                LOG.info("Initialising new relaxation jobs...")

                num_remaining = matador.compute.reset_job_folder()

                # check if we can even finish this generation
                if num_remaining < self.population - len(self.next_gen):
                    LOG.warning(
                        "There were too many failures, not enough remaining calculations to reach target."
                    )
                    LOG.warning(
                        "Consider restarting with a larger allowed failure_ratio."
                    )
                    raise SystemExit(
                        "Failed to return enough successful structures to continue, exiting..."
                    )

                if self.compute_mode == "slurm":
                    # adjust number of nodes so we don't get stuck in the queue
                    if self.max_num_nodes > num_remaining:
                        LOG.info("Adjusted max num nodes to {}".format(self.max_num_nodes))
                        self.max_num_nodes = self.population - len(self.next_gen)

                    self.slurm_submit_relaxations_and_monitor()

                LOG.info("Exiting monitor...")
                exit(0)

            # otherwise, remove unfinished structures from job file and release control of this generation
            else:
                LOG.info("Found enough structures to continue!".format())
                count = 0
                for doc in unrelaxed_gen:
                    structure = doc["source"][0] + ".res"
                    if os.path.isfile(structure):
                        os.remove(structure)
                        count += 1
                LOG.info("Removed {} structures from job folder.".format(count))
                return

        # otherwise, generate a new unrelaxed generation and submit
        else:
            LOG.info("Initialising new generation...")
            self.write_unrelaxed_generation()
            if self.compute_mode == "slurm":
                self.slurm_submit_relaxations_and_monitor()
            LOG.info("Exiting monitor...")
            exit(0)

    def slurm_submit_relaxations_and_monitor(self):
        """ Prepare and submit the appropriate slurm files.

        """
        LOG.info("Preparing to submit slurm scripts...")
        relax_fname = "{}_relax.job".format(self.run_hash)
        # override jobname with this run's hash to allow for selective job killing
        self.slurm_dict["SLURM_JOB_NAME"] = self.run_hash
        compute_string = "run3 {}".format(self.seed)
        matador.compute.slurm.write_slurm_submission_script(
            relax_fname,
            self.slurm_dict,
            compute_string,
            self.walltime_hrs,
            template=self.slurm_template,
        )
        if self.max_num_nodes > self.max_attempts:
            self.max_num_nodes = self.max_attempts
            LOG.info("Adjusted max num nodes to {}".format(self.max_num_nodes))

        # prepare script to read in results
        monitor_fname = "{}_monitor.job".format(self.run_hash)
        compute_string = "python {} >> ilustrado.out 2>> ilustrado.err".format(
            self.entrypoint
        )
        matador.compute.slurm.write_slurm_submission_script(
            monitor_fname,
            self.slurm_dict,
            compute_string,
            1,
            template=self.slurm_template,
        )
        # submit jobs, if any exceptions, cancel all jobs
        try:
            array_job_id = matador.compute.slurm.submit_slurm_script(
                relax_fname, num_array_tasks=self.max_num_nodes
            )
            LOG.info("Submitted job array: {}".format(array_job_id))
            monitor_job_id = matador.compute.slurm.submit_slurm_script(
                monitor_fname, depend_on_job=array_job_id
            )
            LOG.info("Submitted monitor job: {}".format(monitor_job_id))
        except Exception as exc:
            LOG.error("Something went wrong, trying to cancel all jobs: {}".format(exc))
            output = matador.compute.slurm.scancel_all_matching_jobs(name=self.run_hash)
            LOG.error("scancel output: {}".format(output))
            raise SystemExit("Something went wrong, please check the log file.")

    def continuous_birth(self):
        """ Create new generation and relax "as they come", filling the compute
        resources allocated.

        """

        newborns = []
        procs = []
        # queues is a list of mp.Queues where return values will end up
        queues = []
        if self.nodes is None:
            free_nodes = self.nprocs * [None]
            if isinstance(self.ncores, list):
                free_cores = self.nprocs * [None]
            else:
                free_cores = self.nprocs * [self.ncores]
        else:
            free_nodes = deepcopy(self.nodes)
            if isinstance(self.ncores, list):
                free_cores = deepcopy(self.ncores)
            else:
                free_cores = len(self.nodes) * [self.ncores]
        attempts = 0
        print("Computing generation {}:".format(len(self.generations)))
        print(89 * "─")
        print(
            "{:^25} {:^10} {:^10} {:^10} {:^30}".format(
                "ID", "Formula", "# atoms", "Status", "Mutations"
            )
        )
        print(89 * "─")
        # print any recovered structures that already exist
        if self.next_gen:
            for _, structure in enumerate(self.next_gen):
                print(
                    "{:^25} {:^10} {:^10} {:^10} {:^30}".format(
                        structure["source"][0],
                        get_formula_from_stoich(structure["stoichiometry"]),
                        structure["num_atoms"],
                        "Recovered",
                        ", ".join(structure["mutations"]),
                    )
                )
            self.used_sources = [doc["source"][0] for doc in self.next_gen]
        else:
            self.used_sources = []
        try:
            finished = False
            while attempts < self.max_attempts and not finished:
                # if we've reached the target popn, try to kill remaining processes nicely
                if len(self.next_gen) >= self.population:
                    finished = True
                    # while there are still processes running, try to kill them with kill files
                    # that should end the job at the completion of the next CASTEP run
                    self._kill_all_gently(procs, newborns, queues)

                # are we using all nodes? if not, start some processes
                elif len(procs) < self.nprocs and len(self.next_gen) < self.population:
                    # generate structure
                    newborn = self.birth_new_structure()
                    newborn_id = len(newborns)
                    newborns.append(newborn)
                    # clear up and assess CPU resources
                    node = free_nodes.pop()
                    ncores = free_cores.pop()
                    # actually relax structure (or not, if testing is turned on)
                    # TODO: refactor to be more general
                    if self.emt:
                        from ilustrado.util import AseRelaxation

                        queues.append(mp.Queue())
                        relaxer = AseRelaxation(newborns[-1], queues[-1])

                    else:
                        if self.testing:
                            from ilustrado.util import FakeFullRelaxer as FullRelaxer
                        else:
                            from matador.compute import FullRelaxer

                        queues.append(mp.Queue())
                        relaxer = FullRelaxer(
                            ncores=ncores,
                            nnodes=None,
                            node=node,
                            res=newborns[-1],
                            param_dict=self.param_dict,
                            cell_dict=self.cell_dict,
                            verbosity=1,
                            killcheck=True,
                            reopt=False,
                            executable=self.executable,
                            output_queue=queues[-1],
                            start=False,
                            **self.relaxer_params
                        )
                    # store proc object with structure ID, node name, output queue and number of cores
                    procs.append(
                        NewbornProcess(
                            newborn_id,
                            node,
                            mp.Process(target=relaxer.relax),
                            ncores=ncores,
                        )
                    )
                    procs[-1].process.start()
                    LOG.info(
                        "Initialised relaxation for newborn {} on node {} with {} cores.".format(
                            ", ".join(newborns[-1]["source"]), node, ncores
                        )
                    )

                # are we using all nodes? if so, are they all still running?
                elif (
                    all([proc.process.is_alive() for proc in procs])
                    and len(procs) == self.nprocs
                ):
                    # poll processes every second
                    time.sleep(1)
                # so we were using all nodes, but some have died...
                else:
                    LOG.debug("Suspected at least one dead node")
                    # then find the dead ones, collect their results and
                    # delete them so we're no longer using all nodes
                    found_node = False
                    for ind, proc in enumerate(procs):
                        if not proc.process.is_alive():
                            LOG.debug("Found dead node {}".format(proc.node))
                            try:
                                result = queues[ind].get(timeout=60)
                            except Exception:
                                result = False
                                LOG.warning(
                                    "Node {} failed to write to queue for newborn {}".format(
                                        proc.node,
                                        ", ".join(newborns[proc.newborn_id]["source"]),
                                    )
                                )
                            if isinstance(result, dict):
                                self.scrape_result(result, proc=proc, newborns=newborns)
                            try:
                                procs[ind].process.join(timeout=10)
                                LOG.debug(
                                    "Process {proc.newborn_id} on node {proc.node} died gracefully.".format(
                                        proc=proc
                                    )
                                )
                            except Exception:
                                LOG.warning(
                                    "Process {proc.newborn_id} on node {proc.node} has not died gracefully.".format(
                                        proc=proc
                                    )
                                )
                                procs[ind].process.terminate()

                                LOG.warning(
                                    "Process {proc.newborn_id} on node {proc.node} terminated forcefully.".format(
                                        proc=proc
                                    )
                                )
                            if result is not False:
                                free_nodes.append(proc.newborn_id)
                                free_cores.append(proc.ncores)
                            del procs[ind]
                            del queues[ind]
                            attempts += 1
                            found_node = True
                            break
                        # new_free_nodes, new_free_cores, found_node, extra_attempts = self._collect_from_nodes(
                        # procs, newborns, queues
                        # )
                        # attempts += extra_attempts
                        # if new_free_nodes:
                        # free_nodes.append(new_free_nodes)
                        # free_cores.append(new_free_cores)

                        if not found_node:
                            time.sleep(10)

                        break

        except Exception as exc:
            LOG.warning("Something has gone terribly wrong...")
            LOG.error("Exception caught:", exc_info=True)
            print_exc()
            # clean up on error/interrupt
            if len(procs) > 1:
                self.kill_all(procs)
            raise exc

        LOG.info("No longer breeding structures in this generation.")
        # clean up at end either way
        if len(procs) > 1:
            LOG.info(
                "Trying to kill {} on {} processes.".format(self.executable, len(procs))
            )
            self.kill_all(procs)

        if attempts >= self.max_attempts:
            LOG.warning("Failed to return enough successful structures to continue...")
            print(
                "Failed to return enough successful structures to continue, exiting..."
            )
            exit()

    def enforce_elitism(self):
        """ Add elite structures from previous generations
        to bourgeoisie of current generation, through the merit
        of their ancestors alone.
        """
        # add random elite structures from previous gen
        if self.num_elite <= len(self.generations[-1].bourgeoisie):
            probabilities = (
                np.asarray([doc["fitness"] for doc in self.generations[-1].bourgeoisie])
                + 0.0001
            )
            probabilities /= np.sum(probabilities)
            elites = deepcopy(
                np.random.choice(
                    self.generations[-1].bourgeoisie,
                    self.num_elite,
                    replace=False,
                    p=probabilities,
                )
            )
        else:
            elites = deepcopy(self.generations[-1].bourgeoisie)
            if self.debug:
                for doc in elites:
                    print(
                        "Adding doc {} at {} eV/atom".format(
                            " ".join(doc["text_id"]), doc["hull_distance"]
                        )
                    )

        self.next_gen.set_bourgeoisie(
            elites=elites, best_from_stoich=self.best_from_stoich
        )

        LOG.info("Added elite structures from previous generation to next gen.")
        LOG.info("New length of next gen: {}.".format(len(self.next_gen)))
        LOG.info(
            "New length of bourgeoisie: {}.".format(len(self.next_gen.bourgeoisie))
        )

    def reset_and_dump(self):
        """ Add now complete generation to generation list, reset
        the next_gen variable and write dump files.
        """
        # copy next generation to list of generations
        self.generations.append(copy(self.next_gen))
        # reset next_gen ready for, well, the next gen
        self.next_gen = None
        assert self.generations[-1] is not None
        LOG.info(
            "Added current generation {} to generation list.".format(
                len(self.generations) - 1
            )
        )
        # remove interim dump file and create new ones for populace and bourgeoisie
        self.generations[-1].dump(len(self.generations) - 1)
        self.generations[-1].dump_bourgeoisie(len(self.generations) - 1)
        if os.path.isfile("{}-gencurrent.json".format(self.run_hash)):
            os.remove("{}-gencurrent.json".format(self.run_hash))
        if os.path.isfile("{}-genunrelaxed.json".format(self.run_hash)):
            os.remove("{}-genunrelaxed.json".format(self.run_hash))
        LOG.info(
            "Dumped generation file for generation {}".format(len(self.generations) - 1)
        )

    def birth_new_structure(self):
        """ Generate a new structure from current settings.

        Returns:

            dict: newborn structure to be optimised

        """
        possible_parents = (
            self.generations[-1].populace
            if len(self.generations) == 1
            else self.generations[-1].bourgeoisie
        )
        newborn = adapt(
            possible_parents,
            self.mutation_rate,
            self.crossover_rate,
            mutations=self.mutations,
            max_num_mutations=self.max_num_mutations,
            max_num_atoms=self.max_num_atoms,
            structure_filter=self.structure_filter,
            minsep_dict=self.minsep_dict,
            debug=self.debug,
        )
        newborn_source_id = len(self.next_gen)
        if self.compute_mode == "direct":
            while (
                "{}-GA-{}-{}x{}".format(
                    self.seed, self.run_hash, len(self.generations), newborn_source_id
                )
                in self.used_sources
            ):
                newborn_source_id += 1
            self.used_sources.append(
                "{}-GA-{}-{}x{}".format(
                    self.seed, self.run_hash, len(self.generations), newborn_source_id
                )
            )
        newborn["source"] = [
            "{}-GA-{}-{}x{}".format(
                self.seed, self.run_hash, len(self.generations), newborn_source_id
            )
        ]
        LOG.info(
            "Initialised newborn {} with mutations ({})".format(
                ", ".join(newborn["source"]), ", ".join(newborn["mutations"])
            )
        )
        return newborn

    def scrape_result(self, result, proc=None, newborns=None):
        """ Check process for result and scrape into self.next_gen if successful,
        with duplicate detection if desired. If the optional arguments are provided,
        extra logging info will be found when running in `direct` mode.

        Parameters:
            result (dict): containing output from process

        Keyword Arguments:
            proc (tuple)   : standard process tuple from above,
            newborns (list): of new structures to append result to.

        """
        if self.debug:
            if proc is not None:
                print(proc)
            print(dumps(result, sort_keys=True))
        if result.get("optimised"):
            status = "Relaxed"
            if proc is not None:
                LOG.debug(
                    "Newborn {} successfully optimised".format(
                        ", ".join(newborns[proc.newborn_id]["source"])
                    )
                )
                if result.get("parents") is None:
                    LOG.warning(
                        "Failed to get parents for newborn {}.".format(
                            ", ".join(newborns[proc.newborn_id]["source"])
                        )
                    )
                    result["parents"] = newborns[proc.newborn_id]["parents"]
                    result["mutations"] = newborns[proc.newborn_id]["mutations"]

            result = strip_useless(result)
            dupe = False
            if self.check_dupes:
                dupe = self.is_newborn_dupe(result, extra_pdfs=self.extra_pdfs)
                if dupe:
                    status = "Duplicate"
                    if proc is not None:
                        LOG.debug(
                            "Newborn {} is a duplicate and will not be included.".format(
                                ", ".join(newborns[proc.newborn_id]["source"])
                            )
                        )
                    else:
                        LOG.debug(
                            "Newborn {} is a duplicate and will not be included.".format(
                                result["source"][0]
                            )
                        )
                    with open(self.run_hash + "-dupe.json", "a") as f:
                        dump(result, f, sort_keys=False, indent=2)
            if not dupe:
                self.next_gen.birth(result)
                if proc is not None:
                    LOG.info(
                        "Newborn {} added to next generation.".format(
                            ", ".join(newborns[proc.newborn_id]["source"])
                        )
                    )
                else:
                    LOG.info(
                        "Newborn {} added to next generation.".format(
                            result["source"][0]
                        )
                    )
                LOG.info("Current generation size: {}".format(len(self.next_gen)))
                self.next_gen.dump("current")
                LOG.debug("Dumping json file for interim generation...")
        else:
            status = "Failed"
            result = strip_useless(result)
            with open(self.run_hash + "-failed.json", "a") as f:
                dump(result, f, sort_keys=False, indent=2)
        print(
            "{:^25} {:^10} {:^10} {:^10} {:^30}".format(
                result["source"][0],
                get_formula_from_stoich(result["stoichiometry"]),
                result["num_atoms"],
                status,
                ", ".join(result["mutations"]),
            )
        )

    def kill_all(self, procs):
        """ Loop over processes and kill them all.

        Parameters:
            procs (list): list of :obj:`NewbornProcess` in form documented above.

        """
        for proc in procs:
            if self.nodes is not None:
                sp.run(
                    ["ssh", proc.node, "pkill {}".format(self.executable)],
                    timeout=15,
                    stdout=sp.DEVNULL,
                    shell=False,
                )
            proc.process.terminate()

    def recover(self):
        """ Attempt to recover previous generations from files in cwd
        named '<run_hash>_gen{}.json'.format(gen_idx).
        """
        if not os.path.isfile(("{}-gen0.json").format(self.run_hash)):
            exit("Failed to load run, files missing for {}".format(self.run_hash))
        if (
            os.path.isfile(("{}-gencurrent.json").format(self.run_hash))
            and self.compute_mode != "slurm"
        ):
            incomplete = True
            LOG.info("Found incomplete generation for {}".format(self.run_hash))
        else:
            incomplete = False
        try:
            i = 0
            while os.path.isfile("{}-gen{}.json".format(self.run_hash, i)):
                LOG.info(
                    "Trying to load generation {} from run {}.".format(i, self.run_hash)
                )
                fname = "{}-gen{}.json".format(self.run_hash, i)
                self.generations.append(
                    Generation(
                        self.run_hash,
                        i,
                        self.num_survivors,
                        self.num_accepted,
                        dumpfile=fname,
                        fitness_calculator=None,
                    )
                )
                LOG.info(
                    "Successfully loaded {} structures into generation {} from run {}.".format(
                        len(self.generations[-1]), i, self.run_hash
                    )
                )
                i += 1
            print("Recovered from run {}".format(self.run_hash))
            LOG.info("Successfully loaded run {}.".format(self.run_hash))

        except Exception:
            print_exc()
            LOG.error(
                "Something went wrong when reloading run {}".format(self.run_hash)
            )
            exit("Something went wrong when reloading run {}".format(self.run_hash))

        if not self.generations:
            raise SystemExit("No generations found!")

        for i, _ in enumerate(self.generations):
            if not self.testing:
                if i != 0:
                    removed = self.generations[i].clean()
                    LOG.info(
                        "Removed {} structures from generation {}".format(removed, i)
                    )
            if i == len(self.generations) - 1 and len(self.generations) > 1:
                if self.num_elite <= len(self.generations[-2].bourgeoisie):
                    # generate elites with probability proportional to their fitness, but ensure every p is non-zero
                    probabilities = (
                        np.asarray(
                            [doc["fitness"] for doc in self.generations[-2].bourgeoisie]
                        )
                        + 0.0001
                    )
                    probabilities /= np.sum(probabilities)
                    elites = deepcopy(
                        np.random.choice(
                            self.generations[-2].bourgeoisie,
                            self.num_elite,
                            replace=False,
                            p=probabilities,
                        )
                    )
                else:
                    elites = deepcopy(self.generations[-2].bourgeoisie)
                self.generations[i].set_bourgeoisie(
                    best_from_stoich=self.best_from_stoich, elites=elites
                )
            else:
                bourge_fname = "{}-gen{}-bourgeoisie.json".format(self.run_hash, i)
                if os.path.isfile(bourge_fname):
                    self.generations[i].load_bourgeoisie(bourge_fname)
                else:
                    self.generations[i].set_bourgeoisie(
                        best_from_stoich=self.best_from_stoich
                    )
            LOG.info(
                "Bourgeoisie contains {} structures: generation {}".format(
                    len(self.generations[i].bourgeoisie), i
                )
            )
            assert len(self.generations[i]) >= 1
            assert len(self.generations[i].bourgeoisie) >= 1
        if incomplete:
            LOG.info(
                "Trying to load incomplete generation from run {}.".format(
                    self.run_hash
                )
            )
            fname = "{}-gen{}.json".format(self.run_hash, "current")
            self.next_gen = Generation(
                self.run_hash,
                len(self.generations),
                self.num_survivors,
                self.num_accepted,
                dumpfile=fname,
                fitness_calculator=self.fitness_calculator,
            )
            LOG.info(
                "Successfully loaded {} structures into current generation ({}) from run {}.".format(
                    len(self.next_gen), len(self.generations), self.run_hash
                )
            )
            assert len(self.next_gen) >= 1

    def seed_generation_0(self, gene_pool):
        """ Set up first generation from gene pool.

        Parameters:
            gene_pool (list(dict)): list of structure with which to seed generation.

        """

        self.gene_pool = gene_pool

        for ind, parent in enumerate(self.gene_pool):
            if "_id" in parent:
                del self.gene_pool[ind]["_id"]

        # check gene pool is sensible
        errors = []
        if not isinstance(self.gene_pool, list):
            errors.append("Initial gene pool not a list: {}".format(self.gene_pool))
        if not len(self.gene_pool) >= 1:
            errors.append(
                "Initial gene pool not long enough: {}".format(self.gene_pool)
            )
        if errors:
            raise SystemExit("Initial genee pool is not sensible: \n".join(errors))

        generation = Generation(
            self.run_hash,
            0,
            len(gene_pool),
            len(gene_pool),
            fitness_calculator=self.fitness_calculator,
            populace=self.gene_pool,
        )

        generation.rank()
        generation.set_bourgeoisie(best_from_stoich=False)

        LOG.info(
            "Successfully initialised generation 0 with {} members".format(
                len(generation)
            )
        )
        generation.dump(0)
        generation.dump_bourgeoisie(0)

        print(generation)
        self.generations.append(generation)

    def is_newborn_dupe(self, newborn, extra_pdfs=None):
        """ Check each generation for a duplicate structure to the current newborn,
        using PDF calculator from matador.

        Parameters:
            newborn (dict): new structure to screen against the existing,

        Keyword Arguments:
            extra_pdfs (list(dict)): any extra PDFs to compare to, e.g. other hull structures
                not used to seed any generation

        Returns:
            bool: True if duplicate, else False.

        """
        for ind, gen in enumerate(self.generations):
            if ind == 0:
                if gen.is_dupe(newborn, extra_pdfs=extra_pdfs):
                    return True
            else:
                if gen.is_dupe(newborn):
                    return True
        return False

    def finalise_files_for_export(self):
        """ Move unique structures from gen1 onwards to folder "<run_hash>-results". """
        path = "{}-results".format(self.run_hash)
        os.makedirs(path.format(self.run_hash), exist_ok=True)
        LOG.info("Moving unique files to {}-results/...".format(self.run_hash))
        cursor = [struc for gen in self.generations[1:] for struc in gen]
        uniq_inds, _, _, _, = get_uniq_cursor(cursor, projected=True)
        cursor = [cursor[ind] for ind in uniq_inds]
        for doc in cursor:
            source = get_root_source(doc)
            if not source:
                LOG.warning("Issue writing {}".format(doc["source"]))
                continue
            else:
                doc2res(
                    doc, "{}/{}".format(path, source), overwrite=False, hash_dupe=False
                )
            if os.path.isfile("completed/{}".format(source.replace(".res", ".castep"))):
                shutil.copy(
                    "completed/{}".format(source.replace(".res", ".castep")),
                    "{}/{}".format(path, source.replace(".res", ".castep")),
                )

    def _kill_all_gently(self, procs, newborns, queues):
        """ Kill all running processes.

        Parameters:
            procs (list): list of `:obj:NewbornProcess` objects.
            newborns (list): list of corresponding structures.
            queues (list): list of queues that were collecting results.

        """
        kill_attempts = 0
        while procs and kill_attempts < 5:
            for ind, proc in enumerate(procs):
                # create kill file so that matador will stop next finished CASTEP
                filename = "{}.kill".format(newborns[proc.newborn_id]["source"][0])
                with open(filename, "w"):
                    pass
                # wait 1 minute for CASTEP run
                if proc.process.join(timeout=60) is not None:
                    result = queues[ind].get(timeout=60)
                    if isinstance(result, dict):
                        self.scrape_result(result, proc=proc, newborns=newborns)
                    del procs[ind]
                kill_attempts += 1
        if kill_attempts >= 5:
            for ind, proc in enumerate(procs):
                proc.process.terminate()
                del procs[ind]
