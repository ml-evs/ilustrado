""" This file implements the GA algorithm and acts as main(). """
# ilustrado modules
from .adapt import adapt
from .generation import Generation
from .fitness import FitnessCalculator
from .util import strip_useless
from pkg_resources import require
__version__ = require('ilustrado')[0].version
# matador modules
from matador.scrapers.castep_scrapers import res2dict, castep2dict, cell2dict, param2dict
from matador.export import generate_hash, doc2res
from matador.similarity.similarity import get_uniq_cursor
from matador.similarity.pdf_similarity import PDFFactory
from matador.utils.chem_utils import get_formula_from_stoich
from matador.hull import QueryConvexHull
# external libraries
import numpy as np
# standard library
import multiprocessing as mp
import subprocess as sp
import logging
import glob
from os import listdir, makedirs, remove
from os.path import isfile
from time import sleep
from traceback import print_exc
from json import dumps, dump
from sys import exit
from copy import deepcopy, copy


class ArtificialSelector(object):
    """ ArtificialSelector takes an initial gene pool
    and applies a genetic algorithm to optimise some
    fitness function.

    Args:

        | gene_pool         : list(dict), initial cursor to use as "Generation 0",
        | seed              : str, seed name of cell and param files for CASTEP,
        | fitness_metric    : str, currently either 'hull' or 'test',
        | hull              : QueryConvexHull, matador QueryConvexHull object to calculate distances,
        | res_path          : str, path to folder of res files to create hull, if no hull object passed
        | mutation_rate     : float, rate at which to perform single-parent mutations (DEFAULT: 0.5)
        | crossover_rate    : float, rate at which to perform crossovers (DEFAULT: 0.5)
        | num_generations   : int, number of generations to breed before quitting (DEFAULT: 5)
        | num_survivors     : int, number of structures to survive to next generation for breeding
                              (DEFAULT: 10)
        | population        : int, number of structures to breed in any given generation
                              (DEFAULT: 25)
        | failure_ratio     : int, maximum number of attempts per success (DEFAULT: 5)
        | elitism           : float, fraction of next generation to be comprised of elite
                              structures from previous generation (DEFAULT: 0.2)
        | best_from_stoich  : bool, whether to always include the best structure from a
                              stoichiomtery in the next generation,
        | mutations         : list(str), list of mutation names to use,
        | structure_filter  : fn(doc), any function that takes a matador doc and returns True
                              or False,
        | check_dupes       : int, 0 (no checking), 1 (check relaxed structure only), 2 (check
                              unrelaxed mutant) [NOT YET IMPLEMENTED]
        | check_dupes_hull  : bool, compare pdf with all hull structures,
        | max_num_mutations : int, maximum number of mutations to perform on a single structure,
        | max_num_atoms     : int, most atoms allowed in a structure post-mutation/crossover,
        | nodes             : list(str), list of node names to run on,
        | ncores            : int or list(int) specifying the number of cores used by <nodes> per thread,
        | nprocs            : int, total number of processes,
        | recover_from      : str, recover from previous run_hash, by default ilustrado will recover
                              if it finds only one run hash in the folder
        | load_only         : bool, only load structures, do not continue breeding (DEFAULT: False)
        | executable        : str, path to DFT binary (DEFAULT: castep)
        | compute_mode      : str, either `direct` or `slurm` (DEFAULT: direct)
        | max_num_nodes     : int, amount of array jobs to run per generation in `slurm` mode,
        | walltime_hrs      : int, maximum walltime for a SLURM array job,
        | slurm_template    : str, path to template slurm script that includes module loads etc,
        | entrypoint        : str, path to script that initialised this object, such that it can
                              be called by SLURM
        | debug             : bool, printing level,
        | testing           : bool, run test code only if true,
        | verbosity         : int, extra printing level,
        | loglevel          : str, follows std library logging levels.

    """
    def __init__(self, **kwargs):
        """ This is the main entrypoint. Initialises parameters,
        gene pool and begins the GA.
        """
        prop_defaults = {
            # important, required parameters
            'gene_pool': None, 'seed': None, 'fitness_metric': 'hull', 'hull': None, 'res_path': None,
            # recovery and loading parameters
            'recover_from': None, 'load_only': False,
            # GA numerical parameters
            'mutation_rate': 1.0, 'crossover_rate': 0.0, 'num_generations': 5, 'num_survivors': 10,
            'population': 25, 'elitism': 0.2, 'max_num_mutations': 3, 'max_num_atoms': 30,
            # other GA options
            'best_from_stoich': True, 'mutations': None, 'structure_filter': None,
            'check_dupes': 1, 'check_dupes_hull': True, 'failure_ratio': 5,
            # logistical and compute parameters
            'compute_mode': 'direct', 'nodes': None, 'ncores': None, 'nprocs': 1, 'relaxer_params': None,
            'executable': 'castep', 'max_num_nodes': None, 'walltime_hrs': None, 'slurm_template': None,
            'entrypoint': None,
            # debug and logging parameters
            'debug': False, 'testing': False, 'verbosity': 0, 'loglevel': 'info'
        }

        # cache current params to reload again later
        self.current_params = deepcopy(prop_defaults)
        self.current_params.update(kwargs)

        self.__dict__.update(prop_defaults)
        self.__dict__.update(kwargs)

        splash_screen = (r"   _  _              _                     _" + '\n'
                         r"  (_)| |            | |                   | |" + '\n'
                         r"   _ | | _   _  ___ | |_  _ __   __ _   __| |  ___" + '\n'
                         r"  | || || | | |/ __|| __|| '__| / _` | / _` | / _ \ " + '\n'
                         r"  | || || |_| |\__ \| |_ | |   | (_| || (_| || (_) |" + '\n'
                         r"  |_||_| \__,_||___/ \__||_|    \__,_| \__,_| \___/" + '\n\n'
                         "****************************************************\n")
        print('\033[92m\033[1m')
        print('\n' + splash_screen)
        print('\033[0m')

        print('Loading harsh realities of life...', end=' ')

        # post-load checks
        if self.relaxer_params is None:
            self.relaxer_params = dict()
        self.next_gen = None
        if isinstance(self.ncores, list):
            assert len(self.ncores) == len(self.nodes)

        # set up computing resource
        assert self.compute_mode in ['slurm', 'direct']
        if self.compute_mode is 'slurm':
            assert isinstance(self.walltime_hrs, int)
            assert self.walltime_hrs > 0
            assert isinstance(self.max_num_nodes, int)
            assert self.max_num_nodes > 0
            assert isinstance(self.slurm_template, str)
            assert isfile(self.slurm_template)

        elif self.compute_mode is 'direct':
            if self.nodes is not None:
                if self.nprocs != len(self.nodes):
                    logging.warning('Specified procs {} being replaced by number of nodes {}'.format(self.nprocs, len(self.nodes)))
                    self.nprocs = len(self.nodes)

        # set up GA logistics
        self.run_hash = generate_hash()
        self.generations = []  # list to store all generations
        self.num_elite = int(self.elitism * self.num_survivors)
        self.num_accepted = self.num_survivors - self.num_elite
        self.max_attempts = self.failure_ratio * self.population
        assert self.num_survivors < self.population + self.num_elite, 'Survivors > population!'
        assert self.num_accepted < self.population, 'Accepted > population!'
        if self.mutations is not None and isinstance(self.mutations, str):
            self.mutations = [self.mutations]
        assert isinstance(self.max_num_mutations, int)
        assert isinstance(self.max_num_atoms, int)

        # recover from specified run
        if self.recover_from is not None:
            if isinstance(self.recover_from, str):
                self.run_hash = self.recover_from.split('/')[-1]
        # try to look for gen0 files, if multiple are found, safely exit
        else:
            gen0_files = glob.glob('*gen0.json')
            if len(gen0_files) > 1:
                exit('Several incomplete runs found in this folder, please tidy up before re-running.')
            elif len(gen0_files) == 1:
                self.run_hash = gen0_files[0].split('/')[-1].replace('-gen0.json', '')
                self.recover_from = self.run_hash
            else:
                print('No recovery possible, starting fresh run.')

        # set up logging
        numeric_loglevel = getattr(logging, self.loglevel.upper(), None)
        if not isinstance(numeric_loglevel, int):
            exit(self.loglevel, 'is an invalid log level, please use either info, debug or warning.')
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                            filename=self.run_hash+'.log',
                            level=numeric_loglevel)
        logging.info('Starting up ilustrado {}'.format(__version__))

        # initialise fitness calculator
        if self.fitness_metric == 'hull' and self.hull is None:
            if self.res_path is not None and isfile(self.res_path):
                res_files = glob.glob('{}/*.res'.format(self.res_path))
                if len(res_files) == 0:
                    exit('No structures found in {}'.format(self.res_path))
                self.cursor = []
                for res in res_files:
                    self.cursor.append(res2dict(res))
                self.hull = QueryConvexHull(cursor=self.cursor)
            exit('Need to pass a QueryConvexHull object to use hull distance metric.')
        if self.fitness_metric in ['dummy', 'hull_test']:
            self.testing = True
        print('Done!')
        self.fitness_calculator = FitnessCalculator(fitness_metric=self.fitness_metric,
                                                    hull=self.hull, debug=self.debug)
        logging.debug('Successfully initialised fitness calculator.')

        # if we're checking hull pdfs too, make this list now
        if self.check_dupes_hull:
            print('Computing extra PDFs from hull...')
            PDFFactory(self.hull.cursor)
            self.extra_pdfs = [doc['pdf'] for doc in self.hull.cursor]
            # remove pdf object from cursor so generation can be serialized
            for ind, doc in enumerate(self.hull.cursor):
                del self.hull.cursor[ind]['pdf']
        else:
            self.extra_pdfs = None
        assert self.check_dupes in [0, 1, 2]
        logging.info('Successfully initialised similarity lists.')

        if self.recover_from is not None:
            print('Attempting to recover from run {}'.format(self.run_hash))
            if isinstance(self.recover_from, str):
                logging.info('Attempting to recover from previous run {}'.format(self.run_hash))
            self.recover()

        if not self.load_only:
            self.start()

    def start(self):
        """ Start running GA. """
        print('Initialising quantum mechanics...', end=' ')
        # read parameters for relaxation from seed files
        if self.seed is not None:
            seed = self.seed
            self.cell_dict, success_cell = cell2dict(seed, db=False)
            if not success_cell:
                print(self.cell_dict)
                exit('Failed to read cell file.')
            self.param_dict, success_param = param2dict(seed, db=False)
            if not success_param:
                print(self.param_dict)
                exit('Failed to read param file.')
        elif not self.testing:
            exit('Not in testing mode, and failed to provide seed... exiting.')
        else:
            self.seed = 'ga_test'
        print('Done!\n')
        logging.debug('Successfully initialised cell and param files.')

        if self.recover_from is None:
            self.seed_generation_0(self.gene_pool)

        if self.debug:
            print(self.nodes)
        if self.nodes is not None:
            logging.debug('Running on nodes: {}'.format(' '.join(self.nodes)))
        elif self.compute_mode == 'slurm':
            logging.debug('Running through SLURM queue')
        else:
            logging.debug('Running on local machine')

        if self.debug:
            print('Current number of generations: {}. Target number: {}'.format(len(self.generations), self.num_generations))
        # run GA self.num_generations
        while len(self.generations) < self.num_generations:
            self.breed_generation()
            logging.info('Successfully bred generation {}'.format(len(self.generations)))

        assert len(self.generations) == self.num_generations
        print('Reached target number of generations!')
        print('Completed GA!')
        logging.info('Reached target number of generations!')
        logging.info('Completed GA!')
        if not self.testing:
            self.finalise_files_for_export()

    def breed_generation(self):
        """ Build next generation from mutations/crossover of current and
        perform relaxations if necessary.
        """
        # initialise next_gen
        if self.next_gen is None:
            self.next_gen = Generation(self.run_hash,
                                       len(self.generations),
                                       self.num_survivors,
                                       self.num_accepted,
                                       fitness_calculator=self.fitness_calculator)

        # newborns is a list of structures, initially raw then relaxed
        if self.compute_mode == 'direct':
            self.continuous_birth()
        elif self.compute_mode == 'slurm':
            self.batch_birth()

        if len(self.next_gen) < self.population:
            logging.warning('Next gen is smaller than desired population.')
        assert len(self.next_gen) >= self.population

        self.next_gen.rank()
        logging.info('Ranked structures in generation {}'.format(len(self.generations)))
        if not self.testing:
            cleaned = self.next_gen.clean()
            logging.info('Cleaned structures in generation {}, removed {}'.format(len(self.generations), cleaned))

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
            doc2res(newborn, newborn['source'][0], info=False)
            self.next_gen.dump('unrelaxed')

    def batch_birth(self):
        """ Assess whether a generation has been relaxed already. This is done by
        checking for the existence of a file called <run_hash>-genunrelaxed.json.

        If so, match the relaxations up with the cached unrelaxed structures
        and rank them ready for the next generation.

        If not, create a new generation of structures, dump the unrelaxed structures to file,
        create the jobscripts to relax them, submit them and the job to check up on the relaxations,
        then exit.
        """

        entry = 'Beginning birthing of generation {}...'.format(len(self.generations))
        logging.info(entry)
        print(entry)
        fname = '{}-genunrelaxed.json'.format(self.run_hash)
        if isfile(fname):
            logging.info('Found existing generation to be relaxed...')
            # load the unrelaxed structures into a dummy generation
            assert isfile(fname)
            unrelaxed_gen = Generation(self.run_hash,
                                       len(self.generations),
                                       self.num_survivors,
                                       self.num_accepted,
                                       dumpfile=fname,
                                       fitness_calculator=None)
            # check to see which unrelaxed structures completed successfully
            logging.info('Scanning for completed relaxations...')
            for ind, newborn in enumerate(unrelaxed_gen):
                completed_filename = 'completed/{}.castep'.format(newborn['source'][0])
                if isfile(completed_filename):
                    doc, s = castep2dict(completed_filename, db=True)
                    # if all was a success, then "birth" the structure, after checking for uniqueness
                    if s and isinstance(doc, dict):
                        newborn = strip_useless(newborn)
                        doc = strip_useless(doc)
                        newborn.update(doc)
                        assert newborn.get('parents') is not None
                        self.scrape_result(newborn)

            # if there are not enough unrelaxed structures after that run, clean up then resubmit
            logging.info('Found enough {} structures of target {}'.format(len(self.next_gen), self.population))
            if len(self.next_gen) < self.population:
                logging.info('Initialising new relaxation jobs...')
                import matador.slurm
                from matador.compute import reset_job_folder_and_count_remaining
                slurm_dict = matador.slurm.get_slurm_env(fail_loudly=True)
                num_remaining = reset_job_folder_and_count_remaining()

                # check if we can even finish this generation
                if num_remaining < self.population - len(self.next_gen):
                    logging.warning('There were too many failures, not enough remaining calculations to reach target.')
                    logging.warning('Consider restarting with a larger allowed failure_ratio.')
                    exit('Failed to return enough successful structures to continue, exiting...')

                # adjust number of nodes so we don't get stuck in the queue
                if self.max_num_nodes > num_remaining:
                    logging.info('Adjusted max num nodes to {}'.format(self.max_num_nodes))
                    self.max_num_nodes = self.population - len(self.next_gen)

                self.slurm_submit_relaxations_and_monitor(slurm_dict)
                logging.info('Exiting monitor...')
                exit(0)

            # otherwise, remove unfinished structures from job file and release control of this generation
            else:
                logging.info('Found enough structures to continue!'.format())
                completed_filename = 'completed/{}.castep'.format(newborn['source'][0])
                count = 0
                for doc in unrelaxed_gen:
                    structure = doc['source'][0] + '.res'
                    if isfile(structure):
                        remove(structure)
                        count += 1
                logging.info('Removed {} structures from job folder.'.format(count))
                return

        # otherwise, generate a new unrelaxed generation and submit
        else:
            logging.info('Initialising new generation...')
            import matador.slurm
            slurm_dict = matador.slurm.get_slurm_env(fail_loudly=True)
            self.write_unrelaxed_generation()
            self.slurm_submit_relaxations_and_monitor(slurm_dict)
            logging.info('Exiting monitor...')
            exit(0)

    def slurm_submit_relaxations_and_monitor(self, slurm_dict):
        """ Prepare and submit the appropriate slurm files.

        Input:

            | slurm_dict: dict, dict containing SLURM environment variables.

        """
        # prepare script to relax this generation
        import matador.slurm
        logging.info('Preparing to submit slurm scripts...')
        relax_fname = '{}_relax.job'.format(self.run_hash)
        # override jobname with this run's hash to allow for selective job killing
        slurm_dict['SLURM_JOB_NAME'] = self.run_hash
        compute_string = 'run3 {}'.format(self.seed)
        matador.slurm.write_slurm_submission_script(relax_fname,
                                                    slurm_dict,
                                                    compute_string,
                                                    self.walltime_hrs,
                                                    template=self.slurm_template,
                                                    num_nodes=1)
        if self.max_num_nodes > self.max_attempts:
            self.max_num_nodes = self.max_attempts
            logging.info('Adjusted max num nodes to {}'.format(self.max_num_nodes))

        # prepare script to read in results
        monitor_fname = '{}_monitor.job'.format(self.run_hash)
        compute_string = 'python {} >> ilustrado.out 2>> ilustrado.err'.format(self.entrypoint)
        matador.slurm.write_slurm_submission_script(monitor_fname,
                                                    slurm_dict,
                                                    compute_string,
                                                    1,
                                                    template=self.slurm_template,
                                                    num_nodes=1)
        # submit jobs, if any exceptions, cancel all jobs
        try:
            array_job_id = matador.slurm.submit_slurm_script(relax_fname, num_array_tasks=self.max_num_nodes)
            logging.info('Submitted job array: {}'.format(array_job_id))
            monitor_job_id = matador.slurm.submit_slurm_script(monitor_fname, depend_on_job=array_job_id)
            logging.info('Submitted monitor job: {}'.format(monitor_job_id))
        except:
            logging.error('Something went wrong, trying to cancel all jobs.')
            output = matador.slurm.scancel_all_matching_jobs(name=self.run_hash)
            logging.error('scancel output: {}'.format(output))
            exit('Something went wrong, please check the log file.')

    def continuous_birth(self):
        """ Create new generation and relax "as they come", filling the compute
        resources allocated.
        """
        newborns = []
        # procs is a list of tuples [(newborn_id, node, proc), ...]
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
        print('Computing generation {}:'.format(len(self.generations)))
        print(89*'─')
        print('{:^25} {:^10} {:^10} {:^10} {:^30}'.format('ID', 'Formula', '# atoms', 'Status', 'Mutations'))
        print(89*'─')
        # print any recovered structures that already exist
        if len(self.next_gen) > 0:
            for ind, structure in enumerate(self.next_gen):
                print('{:^25} {:^10} {:^10} {:^10} {:^30}'
                      .format(structure['source'][0],
                              get_formula_from_stoich(structure['stoichiometry']),
                              structure['num_atoms'],
                              'Recovered',
                              ', '.join(structure['mutations'])))
            self.used_sources = [doc['source'][0] for doc in self.next_gen]
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
                    kill_attempts = 0
                    while len(procs) > 0 and kill_attempts < 5:
                        for ind, proc in enumerate(procs):
                            # create kill file so that matador will stop next finished CASTEP
                            with open('{}.kill'.format(newborns[proc[0]]['source'][0]), 'w'):
                                pass
                            # wait 1 minute for CASTEP run
                            if proc[2].join(timeout=60) is not None:
                                result = queues[ind].get(timeout=60)
                                if isinstance(result, dict):
                                    self.scrape_result(result, proc=proc, newborns=newborns)
                                del procs[ind]
                            kill_attempts += 1
                    if kill_attempts >= 5:
                        for ind, proc in enumerate(procs):
                            proc[2].terminate()
                            del procs[ind]

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
                    if self.testing:
                        from ilustrado.util import FakeFullRelaxer as FullRelaxer
                    else:
                        from matador.compute import FullRelaxer

                    relaxer = FullRelaxer(ncores=ncores, nnodes=None, node=node,
                                          res=newborns[-1], param_dict=self.param_dict, cell_dict=self.cell_dict,
                                          debug=False, verbosity=self.verbosity, killcheck=True,
                                          reopt=False, executable=self.executable,
                                          start=False, **self.relaxer_params)
                    queues.append(mp.Queue())
                    # store proc object with structure ID, node name, output queue and number of cores
                    procs.append((newborn_id, node,
                                  mp.Process(target=relaxer.relax,
                                             args=(queues[-1],)),
                                  ncores))
                    procs[-1][2].start()
                    logging.info('Initialised relaxation for newborn {} on node {} with {} cores.'
                                 .format(', '.join(newborns[-1]['source']), node, ncores))

                # are we using all nodes? if so, are they all still running?
                elif len(procs) == self.nprocs and all([proc[2].is_alive() for proc in procs]):
                    # poll processes every 10 seconds
                    sleep(10)
                # so we were using all nodes, but some have died...
                else:
                    logging.debug('Suspected at least one dead node')
                    # then find the dead ones, collect their results and
                    # delete them so we're no longer using all nodes
                    found_node = False
                    for ind, proc in enumerate(procs):
                        if not proc[2].is_alive():
                            logging.debug('Found dead node {}'.format(proc[1]))
                            try:
                                result = queues[ind].get(timeout=60)
                            except:
                                result = False
                                logging.warning('Node {} failed to write to queue for newborn {}'
                                                .format(proc[1],
                                                        ', '.join(newborns[proc[0]]['source'])))
                            if isinstance(result, dict):
                                self.scrape_result(result, proc=proc, newborns=newborns)
                            try:
                                procs[ind][2].join(timeout=10)
                                logging.debug('Process {} on node {} died gracefully.'
                                              .format(proc[0], proc[1]))
                            except:
                                logging.warning('Process {} on node {} has not died gracefully.'
                                                .format(proc[0], proc[1]))
                                procs[ind][2].terminate()

                                logging.warning('Process {} on node {} terminated forcefully.'
                                                .format(proc[0], proc[1]))
                            if result is not False:
                                free_nodes.append(proc[1])
                                free_cores.append(proc[3])
                            del procs[ind]
                            del queues[ind]
                            attempts += 1
                            found_node = True
                            # break so that sometimes we skip some cycles of the while loop,
                            # but don't end up oversubmitting
                            break
                    if not found_node:
                        sleep(10)
        except:
            logging.warning('Something has gone terribly wrong...')
            logging.error('Exception caught:', exc_info=True)
            print_exc()
            # clean up on error/interrupt
            if len(procs) > 1:
                self.kill_all(procs)
            raise SystemExit

        logging.info('No longer breeding structures in this generation.')
        # clean up at end either way
        if len(procs) > 1:
            logging.info('Trying to kill {} on {} processes.'.format(self.executable, len(procs)))
            for proc in procs:
                self.kill_all(procs)

        if attempts >= self.max_attempts:
            logging.warning('Failed to return enough successful structures to continue...')
            print('Failed to return enough successful structures to continue, exiting...')
            exit()

    def enforce_elitism(self):
        """ Add elite structures from previous generations
        to bourgeoisie of current generation, through the merit
        of their ancestors alone.
        """
        # add random elite structures from previous gen
        if self.num_elite <= len(self.generations[-1].bourgeoisie):
            probabilities = np.asarray([doc['fitness'] for doc in self.generations[-1].bourgeoisie])+0.0001
            probabilities /= np.sum(probabilities)
            elites = deepcopy(np.random.choice(self.generations[-1].bourgeoisie,
                                               self.num_elite,
                                               replace=False,
                                               p=probabilities))
        else:
            elites = deepcopy(self.generations[-1].bourgeoisie)
            if self.debug:
                for doc in elites:
                    print('Adding doc {} at {} eV/atom'.format(' '.join(doc['text_id']),
                                                               doc['hull_distance']))

        self.next_gen.set_bourgeoisie(elites=elites, best_from_stoich=self.best_from_stoich)

        logging.info('Added elite structures from previous generation to next gen.')
        logging.info('New length of next gen: {}.'.format(len(self.next_gen)))
        logging.info('New length of bourgeoisie: {}.'.format(len(self.next_gen.bourgeoisie)))

    def reset_and_dump(self):
        """ Add now complete generation to generation list, reset
        the next_gen variable and write dump files.
        """
        # copy next generation to list of generations
        self.generations.append(copy(self.next_gen))
        # reset next_gen ready for, well, the next gen
        self.next_gen = None
        assert self.generations[-1] is not None
        logging.info('Added current generation {} to generation list.'
                     .format(len(self.generations)-1))
        # remove interim dump file and create new ones for populace and bourgeoisie
        self.generations[-1].dump(len(self.generations)-1)
        self.generations[-1].dump_bourgeoisie(len(self.generations)-1)
        if isfile('{}-gencurrent.json'.format(self.run_hash)):
            remove('{}-gencurrent.json'.format(self.run_hash))
        if isfile('{}-genunrelaxed.json'.format(self.run_hash)):
            remove('{}-genunrelaxed.json'.format(self.run_hash))
        logging.info('Dumped generation file for generation {}'.format(len(self.generations)-1))

    def birth_new_structure(self):
        """ Generate a new structure from current settings. """
        possible_parents = (self.generations[-1].populace
                            if len(self.generations) == 1
                            else self.generations[-1].bourgeoisie)
        newborn = adapt(possible_parents,
                        self.mutation_rate,
                        self.crossover_rate,
                        mutations=self.mutations,
                        max_num_mutations=self.max_num_mutations,
                        max_num_atoms=self.max_num_atoms,
                        structure_filter=self.structure_filter,
                        debug=self.debug)
        newborn_source_id = len(self.next_gen)
        if self.compute_mode is 'direct':
            while '{}-GA-{}-{}x{}'.format(self.seed, self.run_hash, len(self.generations), newborn_source_id) in self.used_sources:
                newborn_source_id += 1
            self.used_sources.append('{}-GA-{}-{}x{}'.format(self.seed, self.run_hash, len(self.generations), newborn_source_id))
        newborn['source'] = ['{}-GA-{}-{}x{}'.format(self.seed,
                                                     self.run_hash,
                                                     len(self.generations),
                                                     newborn_source_id)]
        logging.info('Initialised newborn {} with mutations ({})'
                     .format(', '.join(newborn['source']),
                             ', '.join(newborn['mutations'])))
        return newborn

    def scrape_result(self, result, proc=None, newborns=None):
        """ Check process for result and scrape into self.next_gen if successful,
        with duplicate detection if desired. If the optional arguments are provided,
        extra logging info will be found when running in `direct` mode.

        Input:

            | result   : dict, containing output from process

        Args:

            | proc     : tuple, standard process tuple from above,
            | newborns : list, of new structures to append result to.

        """
        if self.debug:
            if proc is not None:
                print(proc)
            print(dumps(result, sort_keys=True))
        if result.get('optimised'):
            status = 'Relaxed'
            if proc is not None:
                logging.debug('Newborn {} successfully optimised'
                              .format(', '.join(newborns[proc[0]]['source'])))
                if result.get('parents') is None:
                    logging.warning(
                        'Failed to get parents for newborn {}.'
                        .format(', '.join(newborns[proc[0]]['source'])))
                    result['parents'] = newborns[proc[0]]['parents']
                    result['mutations'] = newborns[proc[0]]['mutations']

            result = strip_useless(result)
            dupe = self.is_newborn_dupe(result, extra_pdfs=self.extra_pdfs)
            if dupe and self.check_dupes in [1, 2]:
                status = 'Duplicate'
                if proc is not None:
                    logging.debug('Newborn {} is a duplicate and will not be included.'
                                  .format(', '.join(newborns[proc[0]]['source'])))
                with open(self.run_hash+'-dupe.json', 'a') as f:
                    dump(result, f, sort_keys=False, indent=2)
            else:
                self.next_gen.birth(result)
                if proc is not None:
                    logging.info('Newborn {} added to next generation.'
                                 .format(', '.join(newborns[proc[0]]['source'])))
                logging.info('Current generation size: {}'
                             .format(len(self.next_gen)))
                self.next_gen.dump('current')
                logging.debug('Dumping json file for interim generation...')
        else:
            status = 'Failed'
            result = strip_useless(result)
            with open(self.run_hash+'-failed.json', 'a') as f:
                dump(result, f, sort_keys=False, indent=2)
        print('{:^25} {:^10} {:^10} {:^10} {:^30}'
              .format(result['source'][0],
                      get_formula_from_stoich(result['stoichiometry']),
                      result['num_atoms'],
                      status,
                      ', '.join(result['mutations'])))

    def kill_all(self, procs):
        """ Loop over processes and kill them all.

        Input:

            | procs: list(tuple), list of processes in form documented above.

        """
        for proc in procs:
            if self.nodes is not None:
                sp.run(['ssh', proc[1], 'pkill {}'.format(self.executable)], timeout=15,
                       stdout=sp.DEVNULL, shell=False)
            proc[2].terminate()

    def recover(self):
        """ Attempt to recover previous generations from files in cwd
        named '<run_hash>_gen{}.json'.format(gen_idx).
        """
        if not isfile(('{}-gen0.json').format(self.run_hash)):
            exit('Failed to load run, files missing for {}'.format(self.run_hash))
        if isfile(('{}-gencurrent.json').format(self.run_hash)) and self.compute_mode != 'slurm':
            incomplete = True
            logging.info('Found incomplete generation for {}'.format(self.run_hash))
        else:
            incomplete = False
        try:
            i = 0
            while isfile('{}-gen{}.json'.format(self.run_hash, i)):
                logging.info('Trying to load generation {} from run {}.'.format(i, self.run_hash))
                fname = '{}-gen{}.json'.format(self.run_hash, i)
                self.generations.append(Generation(self.run_hash,
                                                   i,
                                                   self.num_survivors,
                                                   self.num_accepted,
                                                   dumpfile=fname,
                                                   fitness_calculator=None))
                logging.info('Successfully loaded {} structures into generation {} from run {}.'.format(len(self.generations[-1]), i, self.run_hash))
                i += 1
            print('Recovered from run {}'.format(self.run_hash))
            logging.info('Successfully loaded run {}.'.format(self.run_hash))
        except:
            print_exc()
            logging.error('Something went wrong when reloading run {}'.format(self.run_hash))
            exit('Something went wrong when reloading run {}'.format(self.run_hash))
        assert len(self.generations) > 0
        for i in range(len(self.generations)):
            if not self.testing:
                if i != 0:
                    removed = self.generations[i].clean()
                    logging.info('Removed {} structures from generation {}'.format(removed, i))
            if i == len(self.generations)-1 and len(self.generations) > 1:
                if self.num_elite <= len(self.generations[-2].bourgeoisie):
                    # generate elites with probability proportional to their fitness, but ensure every p is non-zero
                    probabilities = np.asarray([doc['fitness'] for doc in self.generations[-2].bourgeoisie])+0.0001
                    probabilities /= np.sum(probabilities)
                    elites = deepcopy(np.random.choice(self.generations[-2].bourgeoisie,
                                                       self.num_elite,
                                                       replace=False,
                                                       p=probabilities))
                else:
                    elites = deepcopy(self.generations[-2].bourgeoisie)
                self.generations[i].set_bourgeoisie(best_from_stoich=self.best_from_stoich, elites=elites)
            else:
                bourge_fname = '{}-gen{}-bourgeoisie.json'.format(self.run_hash, i)
                if isfile(bourge_fname):
                    self.generations[i].load_bourgeoisie(bourge_fname)
                else:
                    self.generations[i].set_bourgeoisie(best_from_stoich=self.best_from_stoich)
            logging.info('Bourgeoisie contains {} structures: generation {}'.format(len(self.generations[i].bourgeoisie), i))
            assert len(self.generations[i]) >= 1
            assert len(self.generations[i].bourgeoisie) >= 1
        if incomplete:
            logging.info('Trying to load incomplete generation from run {}.'.format(self.run_hash))
            fname = '{}-gen{}.json'.format(self.run_hash, 'current')
            self.next_gen = Generation(self.run_hash,
                                       len(self.generations),
                                       self.num_survivors,
                                       self.num_accepted,
                                       dumpfile=fname,
                                       fitness_calculator=self.fitness_calculator)
            logging.info('Successfully loaded {} structures into current generation ({}) from run {}.'.format(len(self.next_gen),
                                                                                                              len(self.generations),
                                                                                                              self.run_hash))
            assert len(self.next_gen) >= 1

        return

    def seed_generation_0(self, gene_pool):
        """ Set up first generation from gene pool.

        Input:

            gene_pool: list(dict), list of structure with which to seed generation.

        """
        from ilustrado.fitness import default_fitness_function
        # if gene_pool is None, try to read from res files in cwd
        if gene_pool is None:
            res_list = []
            for file in listdir('.'):
                if file.endswith('.res'):
                    res_list.append(file)
            self.gene_pool = []
            for file in res_list:
                doc, s = res2dict(file)
        else:
            # else, expect a list of matador documents
            self.gene_pool = gene_pool
            for ind, parent in enumerate(self.gene_pool):
                if '_id' in parent:
                    del self.gene_pool[ind]['_id']
                if 'hull_distance' in self.gene_pool[ind]:
                    self.gene_pool[ind]['raw_fitness'] = self.gene_pool[ind]['hull_distance']
                else:
                    self.gene_pool[ind]['hull_distance'] = self.fitness_calculator.evaluate([self.gene_pool[ind]])

                fitness = default_fitness_function(self.gene_pool[ind]['raw_fitness'])
                self.gene_pool[ind]['fitness'] = fitness

        # check gene pool is sensible
        try:
            assert isinstance(self.gene_pool, list)
            assert isinstance(self.gene_pool[0], dict)
            assert len(self.gene_pool) >= 1
        except:
            print_exc()
            exit('Initial gene pool is not sensible, exiting...')

        self.generations.append(Generation(self.run_hash,
                                           0,
                                           len(gene_pool),
                                           len(gene_pool),
                                           fitness_calculator=None,
                                           populace=self.gene_pool))

        self.generations[-1].set_bourgeoisie(best_from_stoich=False)

        logging.info('Successfully initialised generation 0 with {} members'
                     .format(len(self.generations[-1])))
        self.generations[0].dump(0)
        self.generations[0].dump_bourgeoisie(0)

        print(self.generations[-1])
        return

    def is_newborn_dupe(self, newborn, extra_pdfs=None):
        """ Check each generation for a duplicate structure to the current newborn,
        using PDF calculator from matador.

        Input:

            | newborn: dict, new structure to screen against the existing,

        Args:

            | extra_pdfs: list(PDF), any extra PDFs to compare to, e.g. other hull structures
                          not used to seed any generation

        Returns:

            | True if duplicate, else False.
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
        path = '{}-results'.format(self.run_hash)
        makedirs(path.format(self.run_hash), exist_ok=True)
        cursor = [struc for gen in self.generations[1:] for struc in gen]
        uniq_inds, _, _, _, = get_uniq_cursor(cursor, projected=True)
        cursor = [cursor[ind] for ind in uniq_inds]
        for doc in cursor:
            source = [src.replace('.castep', '.res') for src in doc['source'] if '-GA-' in src or src.endswith('.castep') or src.endswith('.res')]
            if len(source) == 0:
                print('Issue writing {}'.format(doc['source']))
                continue
            else:
                doc2res(doc, '{}/{}'.format(path, source[0]), overwrite=False, hash_dupe=False)
