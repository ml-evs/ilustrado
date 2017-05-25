# coding: utf-8
""" Some assorted analysis functions. """
import seaborn as sns
import matplotlib.pyplot as plt


def display_gen(generation):
    """ Print some info about the generation. """
    print(generation)


def fitness_swarm_plot(generations, ax=None, save=False):
    """ Make a swarm plot of the fitness of all generations. """
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    sns.set_palette("Dark2", desat=.5)
    if ax is None:
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
    if save:
        plt.savefig(generations[0].run_hash + '_swarmplot.pdf')
    return ax


def plot_new_2d_hull(generations, hull):
    """ Add new structures to old ConvexHull plot. """
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    hull.set_plot_param()
    ax = hull.plot_2d_hull(show=False, ax=ax)
    mutant_colours = plt.cm.Reds_r(np.linspace(0, 0.5, len(generations)))
    crossover_colours = plt.cm.Blues_r(np.linspace(0, 0.5, len(generations)))
    for idx, generation in enumerate(generations[1:]):
        for structure in generation.populace:
            if 'mutations' not in structure:
                colour = mutant_colours[idx]
            else:
                colour = mutant_colours[idx] if 'crossover' not in structure['mutations'] else crossover_colours[idx]
            ax.scatter(structure['concentration'][0], structure['formation_enthalpy_per_atom'],
                       c=colour, s=35, zorder=1000, lw=0)
    plt.savefig(generations[0].run_hash + '_hullplot.pdf')
