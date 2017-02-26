#!/usr/bin/env python

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
from traceback import print_exc
from sys import exit
from copy import deepcopy
import random


class ArtificialSelector(object):
    """ ArtificialSelector takes an initial gene pool
    and applies a genetic algorithm to optimise some
    fitness function.
    """
    def __init__(self, gene_pool=None, seed=None, fitness_metric='dummy', hull=None, debug=False):
        """ Initialise parameters, gene pool and begin GA. """

        splash_screen = ("   _  _              _                     _\n"
                         "  (_)| |            | |                   | |\n"
                         "   _ | | _   _  ___ | |_  _ __   __ _   __| |  ___\n"
                         "  | || || | | |/ __|| __|| '__| / _` | / _` | / _ \ \n"
                         "  | || || |_| |\__ \| |_ | |   | (_| || (_| || (_) |\n"
                         "  |_||_| \__,_||___/ \__||_|    \__,_| \__,_| \___/\n\n"
                         "****************************************************\n")
        print('\033[92m\033[1m', end='')
        print('\n' + splash_screen)
        print('\033[0m')

        print('Loading harsh realities of life...')
        # set GA parameters
        self.population = 15
        self.num_survivors = 10
        self.num_generations = 3
        self.generations = []
        self.hull = hull
        self.fitness_metric = fitness_metric
        self.debug = debug

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

        # hard code these for now
        self.ncores = 16
        self.nnodes = 1

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

    def breed_generation(self):
        """ Build next generation from mutations of current,
        with relaxations.
        """

        next_gen = Generation(fitness_calculator=self.fitness_calculator)

        while len(next_gen) < self.population:
            parent = random.choice(self.generations[-1].bourgeoisie)
            newborn = mutate(parent, debug=self.debug)
            print('Relaxing: {}, {}'.format(newborn['stoichiometry'], newborn['source'][0]))
            print('with mutations:', newborn['mutations'])
            relaxer = FullRelaxer(self.ncores, self.nnodes,
                                  newborn,
                                  self.param_dict, self.cell_dict,
                                  debug=False, verbosity=3)
            if relaxer.success:
                newborn.update(relaxer.result_dict)
                if newborn.get('optimised'):
                    next_gen.birth(newborn)

        next_gen.rank()
        self.generations.append(deepcopy(next_gen))
        self.analyse()

    def analyse(self):
        print('GENERATION {}\n'.format(len(self.generations)))
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
        ax.set_ylabel('Fitness')
        plt.show()
