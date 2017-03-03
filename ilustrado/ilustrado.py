#!/usr/bin/env python
""" This file implements the GA algorithm and acts as main(). """
# ilustrado modules
from .mutate import mutate
from .generation import Generation
from .fitness import FitnessCalculator
# matador modules
from matador.scrapers.castep_scrapers import res2dict, cell2dict, param2dict
from matador.compute import FullRelaxer
# external libraries
import numpy as np
# standard library
from os import listdir
from time import sleep
import multiprocessing as mp
from traceback import print_exc
from sys import exit
from copy import deepcopy
import random


class ArtificialSelector(object):
    """ ArtificialSelector takes an initial gene pool
    and applies a genetic algorithm to optimise some
    fitness function.
    """
    def __init__(self, gene_pool=None, seed=None, fitness_metric='dummy', hull=None,
                 num_generations=5, num_survivors=10, population=25,
                 debug=False, ncores=None, nnodes=2, nodes=None):
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
        self.population = population
        self.num_survivors = num_survivors
        self.num_generations = num_generations
        self.generations = []
        self.hull = hull
        self.fitness_metric = fitness_metric
        self.debug = debug

        # hard code these for now
        self.ncores = ncores
        self.nodes = nodes
        if self.nodes is None:
            self.nnodes = nnodes
        else:
            self.nnodes = len(self.nodes)

        if self.fitness_metric == 'hull' and self.hull is None:
            exit('Need to pass a QueryConvexHull object to use hull distance metric.')
        print('Done!')

        print('Initialising quantum mechanics...')
        # read parameters for relaxation from seed files
        if seed is not None:
            self.cell_dict, success_cell = cell2dict(seed, db=False)
            if not success_cell:
                print(self.cell_dict)
                exit('Failed to read cell file.')
            self.param_dict, success_param = param2dict(seed, db=False)
            if not success_param:
                print(self.param_dict)
                exit('Failed to read param file.')
        print('Done!')

        # initialise fitness calculator
        self.fitness_calculator = FitnessCalculator(fitness_metric=self.fitness_metric,
                                                    hull=self.hull)

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
        self.generations.append(Generation(fitness_calculator=None,
                                           populace=self.gene_pool,
                                           num_survivors=self.num_survivors))

        print(self.generations[-1])

        # run GA self.num_generations
        while len(self.generations) < self.num_generations:
            self.breed_generation()

        # plot simple fitness graph
        self.fitness_swarm_plot()

    def __proc_test(self):
        print('HELLO WORLD, I AM A PROCESS')
        sleep(0.1)
        return

    def breed_generation(self):
        """ Build next generation from mutations of current,
        with relaxations.
        """

        next_gen = Generation(fitness_calculator=self.fitness_calculator)
        # newborns is a list of structures, initially raw then relaxed
        newborns = []
        # procs is a list of tuples [(newborn_id, node, proc), ...]
        procs = []
        if self.nodes is None:
            free_nodes = self.nnodes * [None]
        else:
            free_nodes = self.nodes
        self.max_attempts = 5 * self.population
        attempts = 0
        while len(next_gen) < self.population and attempts < self.max_attempts:
            # are we using all nodes? if not, start some processes
            if len(procs) < self.nnodes:
                parent = random.choice(self.generations[-1].bourgeoisie)
                newborns.append(mutate(parent, debug=self.debug))
                newborn = newborns[-1]
                newborn_id = len(newborns)-1
                node = free_nodes.pop()
                relaxer = FullRelaxer(self.ncores, None, node, newborns[-1], self.param_dict, self.cell_dict, debug=False, verbosity=0, start=False)
                try:
                    print('Relaxing: {}, {}'.format(newborn['stoichiometry'], newborn['source'][0]))
                    print('with mutations:', newborn['mutations'])
                    print('on node {} with {} cores.'.format(node, self.ncores))
                    print(free_nodes, self.nodes)
                except:
                    print_exc()
                    continue
                procs.append((newborn_id, node,
                              mp.Process(target=relaxer.relax)))
                attempts += 1
                procs[-1][2].start()
            # are we using all nodes? if so, are they all still running?
            elif len(procs) == self.nnodes and all([proc[2].is_alive() for proc in procs]):
                # poll processes every second
                sleep(10)
            # so we were using all nodes, but some have died...
            else:
                # then find the dead ones, collect their results and delete them so we're no longer using all nodes
                for ind, proc in enumerate(procs):
                    if not proc[2].is_alive():
                        if newborns[proc[0]].get('optimised'):
                            next_gen.birth(newborns[proc[0]])
                        free_nodes.append(proc[1])
                        procs[ind][2].join()
                        del procs[ind]
                        # break so that sometimes we skip some cycles of the while loop, but don't end up oversubmitting
                        break

        next_gen.rank()
        self.generations.append(deepcopy(next_gen))
        self.analyse()
        self.generations[-1].dump(len(self.generations))

    def analyse(self):
        print('GENERATION {}'.format(len(self.generations)))
        if self.debug:
            print(self.generations[-1])
        print('Most fit: {}'.format(self.generations[-1].most_fit['raw_fitness']))

    def fitness_swarm_plot(self):
        """ Make a swarm plot of the fitness of all generations. """
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_palette("Dark2", desat=.5)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        fitnesses = np.asarray([generation.raw_fitnesses for generation in self.generations if len(generation) > 1]).T
        # sns.violinplot(data=fitnesses, ax=ax, inner=None, color=".6")
        sns.swarmplot(data=fitnesses, ax=ax, linewidth=1, palette=sns.color_palette("Dark2", desat=.5))
        ax.set_xlabel('Generation number')
        ax.set_ylabel('Distance to initial hull (eV/atom)')
        plt.savefig('ga.pdf', dpi=300)
        plt.show()
