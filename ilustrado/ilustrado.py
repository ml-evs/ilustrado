""" This file implements the GA algorithm and acts as main(). """
# ilustrado modules
from .adapt import adapt
from .generation import Generation
from .fitness import FitnessCalculator
from .util import strip_useless
from pkg_resources import require
__version__ = require('ilustrado')[0].version
# matador modules
from matador.scrapers.castep_scrapers import res2dict, cell2dict, param2dict
from matador.compute import FullRelaxer
from matador.export import generate_hash, doc2res
from matador.similarity.similarity import get_uniq_cursor
from matador.similarity.pdf_similarity import PDF
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
        | hull              : QueryConvexHull, matador QueryConvexHull object to calculat distances,
        | res_path          : str, path to folder of res files to create hull, if no hull object passed
        | mutation_rate     : float, rate at which to perform single-parent mutations,
        | crossover_rate    : float, rate at which to perform crossovers,
        | num_generations   : int, number of generations to breed before quitting,
        | num_survivors     : int, number of structures to survive to next generation for breeding,
        | population        : int, number of structures to breed in any given generation,
        | elitism           : float, fraction of next generation to be comprised of elite structures from previous generation,
        | best_from_stoich  : bool, whether to always include the best structure from a stoichiomtery in the next generation,
        | mutations         : list(str), list of mutation names to use,
        | structure_filter  : fn(doc), any function that takes a matador doc and returns True or False,
        | check_dupes       : int, 0 (no checking), 1 (check relaxed structure only), 2 (check unrelaxed mutant) [NOT YET IMPLEMENTED]
        | check_dupes_hull  : bool, compare pdf with all hull structures,
        | max_num_mutations : int, maximum number of mutations to perform on a single structure,
        | max_num_atoms     : int, most atoms allowed in a structure post-mutation/crossover,
        | monitor           : bool, whether or not to restart nodes that fail unexpectedly,
        | nodes             : list(str), list of node names to run on,
        | ncores            : int, or list of integers specifying the number of cores used by <nodes> per thread,
        | nprocs            : int, number of threads to run per node,
        | recover_from      : str, recover from previous run_hash,
        | load_only         : bool, only load structures, do not continue breeding,
        | executable        : str, path to DFT binary,
        | debug             : bool, printing level,
        | testing           : bool, run test code only if true,
        | verbosity         : int, extra printing level,
        | loglevel          : str, follows std library logging levels.

    """
    def __init__(self,
                 gene_pool=None,
                 seed=None,
                 fitness_metric='hull',
                 hull=None,
                 res_path=None,
                 mutation_rate=1.0,
                 crossover_rate=0.0,
                 num_generations=5,
                 num_survivors=10,
                 population=25,
                 elitism=0.2,
                 best_from_stoich=True,
                 mutations=None,
                 structure_filter=None,
                 check_dupes=1,
                 check_dupes_hull=False,
                 max_num_mutations=3,
                 max_num_atoms=40,
                 monitor=False,
                 nodes=None,
                 ncores=None,
                 nprocs=1,
                 recover_from=None,
                 relaxer_params=None,
                 load_only=False,
                 executable='castep',
                 debug=False,
                 testing=False,
                 verbosity=0,
                 loglevel='info'):
        """ Initialise parameters, gene pool and begin GA. """

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
        # set GA parameters
        self.population = population  # target size of each generation
        self.num_survivors = num_survivors
        self.num_generations = num_generations
        self.elitism = elitism  # fraction of previous generation to carry through
        self.num_elite = int(self.elitism * self.num_survivors)
        self.num_accepted = self.num_survivors - self.num_elite
        assert self.num_survivors < self.population + self.num_elite, 'Survivors > population!'
        assert self.num_accepted < self.population, 'Accepted > population!'
        self.best_from_stoich = best_from_stoich
        self.generations = []  # list to store all generations
        self.hull = hull  # QueryConvexHull object to calculate hull fitness
        self.fitness_metric = fitness_metric  # choose method of ranking structures
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutations = mutations
        if relaxer_params is not None:
            self.relaxer_params = relaxer_params
        else:
            self.relaxer_params = dict()
        self.seed = seed
        self.next_gen = None
        self.gene_pool = gene_pool
        self.res_path = res_path
        if self.mutations is not None and isinstance(self.mutations, str):
            self.mutations = [self.mutations]
        self.max_num_mutations = max_num_mutations
        assert isinstance(self.max_num_mutations, int)
        self.max_num_atoms = max_num_atoms
        assert isinstance(self.max_num_atoms, int)
        self.structure_filter = structure_filter
        self.check_dupes = int(check_dupes)
        self.check_dupes_hull = check_dupes_hull
        # if we're checking hull pdfs too, make this list now
        if self.check_dupes_hull:
            print('Computing extra PDFs from hull...')
            import progressbar
            bar = progressbar.ProgressBar(term_width=80)
            self.extra_pdfs = []
            for doc in bar(hull.query.cursor):
                self.extra_pdfs.append(PDF(doc, projected=True))
        else:
            self.extra_pdfs = None
        assert self.check_dupes in [0, 1, 2]
        if self.check_dupes == 2:
            raise NotImplementedError

        # set up logistics
        self.run_hash = generate_hash()
        self.recover_from = recover_from  # recover from previous run with this hash
        self.executable = executable
        self.debug = debug
        self.verbosity = verbosity
        self.testing = testing
        self.ncores = ncores
        self.nprocs = nprocs
        self.nodes = nodes
        if isinstance(self.ncores, list):
            assert len(self.ncores) == len(self.nodes)
        self.monitor = monitor  # bool: if a node dies, leave it dead

        if self.recover_from is not None:
            if isinstance(self.recover_from, str):
                self.run_hash = self.recover_from.split('/')[-1]

        # set up logging
        numeric_loglevel = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_loglevel, int):
            exit(loglevel, 'is an invalid log level, please use either info, debug or warning.')
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                            filename=self.run_hash+'.log',
                            level=numeric_loglevel)
        logging.info('Starting up ilustrado {}'.format(__version__))

        if self.nodes is not None:
            self.nprocs = len(self.nodes)
            if self.nprocs != nprocs and nprocs is not None:
                logging.warning('Specified procs {} being replaced by number of nodes {}'.format(self.nprocs, nprocs))
        self.initial_nodes = nodes

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

        if self.recover_from is not None:
            print('Attempting to recover from run {}'.format(self.run_hash))
            if isinstance(self.recover_from, str):
                logging.info('Attempting to recover from previous run {}'.format(self.run_hash))
            self.recover()

        if not load_only:
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
        if self.next_gen is None:
            self.next_gen = Generation(self.run_hash,
                                       len(self.generations),
                                       self.num_survivors,
                                       self.num_accepted,
                                       fitness_calculator=self.fitness_calculator)
        # newborns is a list of structures, initially raw then relaxed
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
        self.max_attempts = 5 * self.population
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
                                    self.scrape_result(result, proc, newborns)
                                del procs[ind]
                            kill_attempts += 1
                    if kill_attempts >= 5:
                        for ind, proc in enumerate(procs):
                            proc[2].terminate()
                            del procs[ind]

                # are we using all nodes? if not, start some processes
                elif len(procs) < self.nprocs and len(self.next_gen) < self.population:
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
                    newborn_id = len(newborns)
                    newborn_source_id = len(self.next_gen)
                    while '{}-GA-{}-{}x{}'.format(self.seed, self.run_hash, len(self.generations), newborn_source_id) in self.used_sources:
                        newborn_source_id += 1
                    self.used_sources.append('{}-GA-{}-{}x{}'.format(self.seed, self.run_hash, len(self.generations), newborn_source_id))
                    newborn['source'] = ['{}-GA-{}-{}x{}'.format(self.seed,
                                                                 self.run_hash,
                                                                 len(self.generations),
                                                                 newborn_source_id)]
                    newborns.append(newborn)
                    node = free_nodes.pop()
                    ncores = free_cores.pop()
                    logging.info('Initialised newborn {} with mutations ({})'
                                 .format(', '.join(newborns[-1]['source']),
                                         ', '.join(newborns[-1]['mutations'])))
                    if self.testing:
                        from ilustrado.util import FakeFullRelaxer
                        relaxer = FakeFullRelaxer(ncores=ncores, nnodes=None, node=node,
                                                  res=newborns[-1], param_dict=self.param_dict, cell_dict=self.cell_dict,
                                                  debug=False, verbosity=self.verbosity,
                                                  reopt=False, executable=self.executable,
                                                  start=False)
                    else:
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
                                if self.monitor:
                                    logging.warning('Assuming {} has died, removing from node list for duration.'.format(proc[0]))
                                    self.nodes.remove(proc[1])
                                    self.nprocs -= 1
                                if self.monitor and len(self.nodes) == 0:
                                    logging.warning('Number of nodes reached 0, exiting...')
                                    exit('No nodes remain, exiting...')
                            if isinstance(result, dict):
                                self.scrape_result(result, proc, newborns)
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
                            # if monitor is True and the node didn't return a result, then don't use again
                            if not self.monitor or result is not False:
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

        if len(self.next_gen) < self.population:
            logging.warning('Next gen is smaller than desired population.')
        assert len(self.next_gen) >= self.population

        self.next_gen.rank()
        logging.info('Ranked structures in generation {}'.format(len(self.generations)))
        if not self.testing:
            cleaned = self.next_gen.clean()
            logging.info('Cleaned structures in generation {}, removed {}'.format(len(self.generations), cleaned))

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

        self.generations.append(copy(self.next_gen))
        self.next_gen = None
        assert self.generations[-1] is not None
        logging.info('Added current generation {} to generation list.'
                     .format(len(self.generations)-1))
        if isfile('{}-gencurrent.json'.format(self.run_hash)):
            remove('{}-gencurrent.json'.format(self.run_hash))
        self.generations[-1].dump(len(self.generations)-1)
        self.generations[-1].dump_bourgeoisie(len(self.generations)-1)
        logging.info('Dumped generation file for generation {}'.format(len(self.generations)-1))
        print(self.generations[-1])

    def scrape_result(self, result, proc, newborns):
        """ Check process for result and scrape into self.next_gen if successful,
        with duplicate detection if desired.

        Input:

            | result   : dict, containing output from process,
            | proc     : tuple, standard process tuple from above,
            | newborns : list, of new structures to append result to.

        """
        if self.debug:
            print(proc)
            print(dumps(result, sort_keys=True))
        if result.get('optimised'):
            status = 'Relaxed'
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
                logging.debug('Newborn {} is a duplicate and will not be included.'
                              .format(', '.join(newborns[proc[0]]['source'])))
                with open(self.run_hash+'-dupe.json', 'a') as f:
                    dump(result, f, sort_keys=False, indent=2)
            else:
                self.next_gen.birth(result)
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
              .format(newborns[proc[0]]['source'][0],
                      get_formula_from_stoich(
                          newborns[proc[0]]['stoichiometry']),
                      newborns[proc[0]]['num_atoms'],
                      status,
                      ', '.join(newborns[proc[0]]['mutations'])))

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
        if isfile(('{}-gencurrent.json').format(self.run_hash)):
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
