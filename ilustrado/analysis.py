# coding: utf-8
""" Some assorted analysis functions. """


def display_gen(generation):
    """ Print some info about the generation. """
    print(generation)


def fitness_swarm_plot(generations):
    """ Make a swarm plot of the fitness of all generations. """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    sns.set_palette("Dark2", desat=.5)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    fitnesses = np.asarray([generation.raw_fitnesses for generation in generations
                            if len(generation) > 1]).T
    sns.violinplot(data=fitnesses, ax=ax, inner=None, color=".6")
    sns.swarmplot(data=fitnesses, ax=ax,
                  linewidth=1,
                  palette=sns.color_palette("Dark2", desat=.5))
    ax.set_xlabel('Generation number')
    ax.set_ylabel('Distance to initial hull (eV/atom)')
    plt.savefig(generations[0].run_hash + '_swarmplot.pdf')
