#!/usr/bin/env python

# ilustrado modules
from mutate import mutate
from fitness import evaluate_fitness
# matador modules
from matador.scrapers.castep_scrapers import res2dict
from matador.utils.chem_utils import get_formula_from_stoich
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
    def __init__(self, gene_pool=None, debug=False):
        """ Initialise parameters, gene pool and begin GA. """

        print('Initialising Mother Nature...')

        self.population = 100
        self.num_survivors = 10
        self.num_generations = 10
        self.generations = []

        self.debug = debug

        # if gene_pool is None, try to read from res files in pwd
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

        # check gene pool is sensible
        try:
            assert isinstance(self.gene_pool, list)
            assert isinstance(self.gene_pool[0], dict)
            assert len(self.gene_pool) >= 1
        except:
            print_exc()
            exit('Initial gene pool is not sensible, exiting...')

        # generation 0 is initial gene pool
        self.generations.append(Generation(self.gene_pool, num_survivors=self.num_survivors))

        if self.debug:
            print('Generation 0 initialised')
            print(self.generations[-1])

        # breed for self.num_generations
        self.breed()

    def breed(self):
        """ Build next generation from mutations of current. """

        num_generations = 0
        while num_generations < self.num_generations:
            next_gen = Generation()

            while len(next_gen) < self.population:
                parent = random.choice(self.generations[-1].bourgeoisie)
                child = mutate(parent, debug=self.debug)
                next_gen.birth(child)

            next_gen.rank()
            self.generations.append(deepcopy(next_gen))
            num_generations += 1
            self.analyse()

        self.fitness_violin_plot()

    def analyse(self):
        print('GENERATION {}\n'.format(len(self.generations)))
        print(self.generations[-1])

    def fitness_violin_plot(self):
        """ Make a violin plot of the fitness of all generations. """
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_palette("Dark2", desat=.5)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fitnesses = np.asarray([generation.fitnesses for generation in self.generations if len(generation) > 1]).T
        sns.violinplot(data=fitnesses, ax=ax, width=0.6, lw=0.5, palette=sns.color_palette("Dark2", desat=.5))
        ax.set_xlabel('Generation number')
        ax.set_ylabel('Fitness')
        plt.show()


class Generation(object):
    """ Stores each generation of structures. """

    def __init__(self, populace=[], num_survivors=10):

        self.populace = populace
        self.num_survivors = num_survivors

    def __len__(self):
        return len(self.populace)

    def __str__(self):
        gen_string = 80*'=' + '\n'
        gen_string += 'Number of members: {}\n'.format(len(self.populace))
        gen_string += 'Number of survivors: {}\n'.format(len(self.bourgeoisie))
        for populum in self.populace:
            gen_string += '{} {}\n'.format(get_formula_from_stoich(populum['stoichiometry']), populum['fitness'])
        gen_string += 80*'=' + '\n'
        return gen_string

    def birth(self, populum):
        self.populace.append(populum)

    def rank(self):
        for ind, populum in enumerate(self.populace):
            self.populace[ind]['fitness'] = evaluate_fitness(populum)

    @property
    def bourgeoisie(self):
        return sorted(self.populace, key=lambda member: member['fitness'])[:self.num_survivors]

    @property
    def fitnesses(self):
        return [populum['fitness'] for populum in self.populace]

    @property
    def most_fit(self):
        assert self.bourgeoisie[-1].fitness == max(self.fitnesses)
        return self.bourgeoisie[-1]

    @property
    def average_pleb_fitness(self):
        population = len(self.populace)
        average_fitness = 0
        for populum in self.populace:
            average_fitness += populum['fitness'] / population
        return average_fitness

    @property
    def average_bourgeois_fitness(self):
        population = len(self.bourgeoisie)
        average_fitness = 0
        for populum in self.bourgeoisie:
            average_fitness += populum['fitness'] / population
        return average_fitness
