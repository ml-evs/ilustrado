#!/usr/bin/env python

# immacolata modules
from mutate import mutate
# matador modules
from matador.scrapers.castep_scrapers import res2dict
# standard library
from os import listdir
from traceback import print_exc
from sys import exit
import random


class ArtificialSelector(object):
    """ ArtificialSelector takes an initial gene pool
    and applies a genetic algorithm to optimise some
    fitness function.
    """
    def __init__(self, gene_pool=None):
        """ Initialise parameters, gene pool and begin GA. """

        self.population = 100
        self.generations = []

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

        # check gene pool is sensible
        try:
            assert isinstance(self.gene_pool, list)
            assert isinstance(self.gene_pool[0], dict)
            assert len(self.gene_pool) >= 1
        except:
            print_exc()
            exit('Initial gene pool is not sensible, exiting...')
        
        # generation 0 is initial gene pool
        self.generations.append(Generation(self.gene_pool))

        self.breed()

    def breed(self):

        next_gen = Generation()
        while len(next_gen) < self.populace_size:
            parent = random.choice(self.generations[-1])
            child = mutate(parent)
            next_gen.birth(child)

        next_gen.rank()

        raise NotImplementedError


class Generation(object):
    """ Stores each generation of structures. """

    def __init__(self, populace=[]):
        self.populace = populace
        self.fitness_ranks = []

    def __len__(self):
        return len(self.populace)

    def birth(self, populum):
        self.populace.append(populum)

    def rank(self):
        for populum in self.populace:
            populum.evaulate_fitness()
        self.fitness_ranks = sorted(self.populace, key=lambda k: k['fitness'])

    def class_warfare(self):
        raise NotImplementedError
