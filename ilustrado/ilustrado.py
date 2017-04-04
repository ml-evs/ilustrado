#!/usr/bin/env python
""" This file implements the GA algorithm and acts as main(). """
# ilustrado modules
from .mutate import mutate
from .generation import Generation
from .fitness import FitnessCalculator
from .analysis import display_gen, fitness_swarm_plot
from .util import strip_useless
# matador modules
from matador.scrapers.castep_scrapers import res2dict, cell2dict, param2dict
from matador.compute import FullRelaxer
from matador.export import generate_hash
# external libraries
import numpy as np
# standard library
from os import listdir
from time import sleep
import multiprocessing as mp
from traceback import print_exc
from json import dumps
from sys import exit
from copy import deepcopy
import random
import logging


class ArtificialSelector(object):
    """ ArtificialSelector takes an initial gene pool
    and applies a genetic algorithm to optimise some
    fitness function.
    """
    def __init__(self, gene_pool=None, seed=None, fitness_metric='hull', hull=None,
                 num_generations=5, num_survivors=10, population=25, elitism=0.2, recover=False,
                 debug=False, ncores=None, nnodes=None, nodes=None, loglevel='info'):
        """ Initialise parameters, gene pool and begin GA. """

        splash_screen = ("   _  _              _                     _\n"
                         "  (_)| |            | |                   | |\n"
                         "   _ | | _   _  ___ | |_  _ __   __ _   __| |  ___\n"
                         "  | || || | | |/ __|| __|| '__| / _` | / _` | / _ \ \n"
                         "  | || || |_| |\__ \| |_ | |   | (_| || (_| || (_) |\n"
                         "  |_||_| \__,_||___/ \__||_|    \__,_| \__,_| \___/\n\n"
                         "****************************************************\n")
        print('\033[92m\033[1m')
        print('\n' + splash_screen)
        print('\033[0m')

        print('Loading harsh realities of life...')
        # set GA parameters
        self.population = population  # target size of each generation
        self.num_survivors = num_survivors  # number of survivors per generation
        self.num_generations = num_generations  # desired number of generations
        self.elitism = elitism  # fraction of previous generation to carry throough
        self.num_elite = int(self.elitism * self.num_survivors)
        assert self.num_survivors < self.population + self.num_elite, 'Survivors > population!'
        self.generations = []  # list to store all generations
        self.hull = hull  # QueryConvexHull object to calculate hull fitness
        self.fitness_metric = fitness_metric  # choose method of ranking structures

        # set up logistics
        self.run_hash = generate_hash()
        self.recover = recover  # recover from previous run
        self.debug = debug
        self.testing = False
        self.ncores = ncores
        self.nodes = nodes
        self.initial_nodes = nodes
        if self.nodes is None:
            # if no nodes specified, run on all cores of current node
            self.nnodes = 1
        else:
            self.nnodes = len(self.nodes)

        # set up logging
        numeric_loglevel = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_loglevel, int):
            exit(loglevel, 'is an invalid log level, please use either info, debug or warning.')
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                            filename=self.run_hash+'.log',
                            level=numeric_loglevel)

        if self.fitness_metric == 'hull' and self.hull is None:
            exit('Need to pass a QueryConvexHull object to use hull distance metric.')
        if self.fitness_metric in ['dummy', 'hull_test']:
            self.testing = True
        print('Done!')

        print('Initialising quantum mechanics...')
        # read parameters for relaxation from seed files
        if seed is not None:
            self.seed = seed
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
        print('Done!')
        logging.debug('Successfully initialised cell and param files.')

        # initialise fitness calculator
        self.fitness_calculator = FitnessCalculator(fitness_metric=self.fitness_metric,
                                                    hull=self.hull, debug=self.debug)
        logging.debug('Successfully initialised fitness calculator.')

        # if gene_pool is None, try to read from res files in cwd
        print('Seeding generation 0')
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
                del self.gene_pool[ind]['_id']
                self.gene_pool[ind]['fitness'] = -1
                self.gene_pool[ind]['raw_fitness'] = self.gene_pool[ind]['hull_distance']

        # check gene pool is sensible
        try:
            assert isinstance(self.gene_pool, list)
            assert isinstance(self.gene_pool[0], dict)
            assert len(self.gene_pool) >= 1
        except:
            print_exc()
            exit('Initial gene pool is not sensible, exiting...')

        # generation 0 is initial gene pool
        self.generations.append(Generation(self.run_hash,
                                           0,
                                           self.num_survivors,
                                           fitness_calculator=None,
                                           populace=self.gene_pool))

        logging.info('Successfully initialised generation 0 with {} members'
                     .format(len(self.generations[-1])))

        print(self.generations[-1])

        if self.debug:
            print(self.nodes)
        logging.debug('Running on nodes: {}'.format(' '.join(self.nodes)))

        # run GA self.num_generations
        while len(self.generations) < self.num_generations:
            self.breed_generation()
            logging.info('Successfully bred generation {}'.format(len(self.generations)))
            fitness_swarm_plot(self.generations)

        logging.info('Completed GA!')
        # plot simple fitness graph
        fitness_swarm_plot(self.generations)

    def __proc_test(self):
        print('HELLO WORLD, I AM A PROCESS')
        sleep(0.1)
        return

    def breed_generation(self):
        """ Build next generation from mutations of current,
        and perform relaxations if necessary.
        """
        next_gen = Generation(self.run_hash,
                              len(self.generations),
                              self.num_survivors,
                              fitness_calculator=self.fitness_calculator)
        # newborns is a list of structures, initially raw then relaxed
        newborns = []
        # procs is a list of tuples [(newborn_id, node, proc), ...]
        procs = []
        # queues is a list of mp.Queues where return values will end up
        queues = []
        if self.nodes is None:
            free_nodes = self.nnodes * [None]
        else:
            free_nodes = self.nodes
        self.max_attempts = 5 * self.population
        attempts = 0
        try:
            while len(next_gen) < self.population and attempts < self.max_attempts:
                # are we using all nodes? if not, start some processes
                if len(procs) < self.nnodes:
                    # if its the first generation use all possible structures as parents
                    if len(self.generations) == 1:
                        parent = random.choice(self.generations[-1].populace)
                    else:
                        parent = random.choice(self.generations[-1].bourgeoisie)
                    newborns.append(mutate(parent, debug=self.debug))
                    # set source and parents of newborn
                    newborns[-1]['source'] = ['{}-GA-{}-{}x{}'.format(self.seed,
                                                                      self.run_hash,
                                                                      len(self.generations),
                                                                      len(newborns))]
                    for source in parent['source']:
                        if source.endswith('.res') or source.endswith('.castep'):
                            parent_source = source.split('/')[-1] \
                                                  .replace('.res', '').replace('.castep', '')
                    newborns[-1]['parents'] = [parent_source]
                    newborn = newborns[-1]
                    newborn_id = len(newborns)-1
                    node = free_nodes.pop()
                    logging.info('Initialised newborn {} with mutations ({})'
                                 .format(', '.join(newborns[-1]['source']),
                                         ', '.join(newborns[-1]['mutations'])))
                    relaxer = FullRelaxer(self.ncores, None, node,
                                          newborns[-1], self.param_dict, self.cell_dict,
                                          debug=False, verbosity=0,
                                          start=False, redirect=False)
                    if self.debug:
                        print('Relaxing: {}, {}'.format(newborn['stoichiometry'],
                                                        newborn['source'][0]))
                        print('with mutations:', newborn['mutations'])
                        print('on node {} with {} cores.'.format(node, self.ncores))
                    queues.append(mp.Queue())
                    procs.append((newborn_id, node,
                                  mp.Process(target=relaxer.relax,
                                             args=(queues[-1],))))
                    procs[-1][2].start()
                    logging.info('Initialised relaxation for newborn {} on node {} with {} cores.'
                                 .format(', '.join(newborns[-1]['source']), node, self.ncores))
                # are we using all nodes? if so, are they all still running?
                elif len(procs) == self.nnodes and all([proc[2].is_alive() for proc in procs]):
                    # poll processes every 10 seconds
                    sleep(10)
                # so we were using all nodes, but some have died...
                else:
                    logging.debug('Suspected at least one dead node')
                    # then find the dead ones, collect their results and
                    # delete them so we're no longer using all nodes
                    for ind, proc in enumerate(procs):
                        if not proc[2].is_alive():
                            logging.debug('Found dead node {}'.format(proc[1]))
                            try:
                                result = queues[ind].get(timeout=60)
                            except:
                                logging.warning('Node {} failed to write to queue for newborn {}'
                                                .format(proc[1],
                                                        ', '.join(newborns[proc[0]]['source'])))
                                result = False
                            if isinstance(result, dict):
                                if self.debug:
                                    print(proc)
                                    print(dumps(result, sort_keys=True))
                                if result.get('optimised'):
                                    logging.debug('Newborn {} successfully optimised'
                                                  .format(', '.join(newborns[proc[0]]['source'])))
                                    if result.get('parents') is None:
                                        logging.debug('Failed to get parents for newborn {}.'
                                                      .format(', '
                                                              .join(newborns[proc[0]]['source'])))
                                        result['parents'] = newborns[proc[0]]['parents']
                                        result['mutations'] = newborns[proc[0]]['mutations']
                                    result = strip_useless(result)
                                    next_gen.birth(result)
                                    logging.info('Newborn {} added to next generation.'
                                                 .format(', '.join(newborns[proc[0]]['source'])))
                                    next_gen.dump('current')
                                    logging.debug('Dumping json file for interim generation...')
                            try:
                                procs[ind][2].join(timeout=10)
                                logging.debug('Process {} on node {} died gracefully.'
                                              .format(ind, proc[1]))
                            except:
                                logging.warning('Process {} on node {} has not died gracefully.'
                                                .format(ind, proc[1]))
                                procs[ind][2].terminate()
                                logging.warning('Process {} on node {} terminated forcefully.'
                                                .format(ind, proc[1]))
                            free_nodes.append(proc[1])
                            del procs[ind]
                            del queues[ind]
                            if self.debug:
                                print(len(newborns), len(next_gen), attempts, self.population)
                            attempts += 1
                            # break so that sometimes we skip some cycles of the while loop,
                            # but don't end up oversubmitting
                            break
        except:
            print('EVERYTHING HAS FALLEN APART.')
            logging.warning('Something has gone terribly wrong, caught exception.')
            logging.error(exc_info=True)
            print_exc()
            # clean up on error/interrupt
            if len(procs) > 1:
                for proc in procs:
                    proc[2].terminate()
            raise SystemExit

        # clean up at end either way
        if len(procs) > 1:
            for proc in procs:
                proc[2].terminate()

        if attempts >= self.max_attempts:
            logging.warning('Failed to return enough successful structures to continue...')
            print('Failed to return enough successful structures to continue, exiting...')
            exit()

        if len(next_gen) < self.population:
            logging.warning('Next gen is smaller than desired population.')
        assert len(next_gen) >= self.population
        # add random elite structures from previous gen
        if self.debug:
            print('Adding {} structures from previous generation...'.format(self.num_elite))
        for i in range(self.num_elite):
            doc = deepcopy(np.random.choice(self.generations[-1].bourgeoisie))
            next_gen.birth(doc)
            if self.debug:
                print('Adding doc {} at {} eV/atom'.format(' '.join(doc['text_id']),
                                                           doc['hull_distance']))
        logging.info('Added elite structures from previous generation to next gen.')
        if self.debug:
            print('New length: {}'.format(len(next_gen)))
        logging.info('New length of next gen: {}.'.format(len(next_gen)))
        next_gen.rank()
        logging.info('Ranked structures in generation {}'.format(len(self.generations)-1))
        self.generations.append(next_gen)
        logging.info('Added current generation {} to generation list.'
                     .format(len(self.generations)-1))
        self.generations[-1].dump(len(self.generations)-1)
        logging.info('Dumped generation file for generation {}'.format(len(self.generations)-1))
        display_gen(self.generations[-1])

    def recover(self):
        """ Attempt to recover previous generations from files in cwd
        named 'gen_{}.json'.format(gen_idx).
        """
        raise NotImplementedError
