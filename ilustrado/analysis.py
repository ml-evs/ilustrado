# coding: utf-8
""" Some assorted analysis functions. """

def fitness_swarm_plot(generations, ax=None, save=False):
    """ Make a swarm plot of the fitness of all generations. """
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
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


def plot_new_2d_hull(generations, hull, points=True, label_hull=True, save=True):
    """ Add new structures to old ConvexHull plot. """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    hull.set_plot_param()
    ax = hull.plot_2d_hull(show=False, ax=ax, plot_points=points, labels=label_hull)
    ax2 = ax.twinx()
    mutant_colours = plt.cm.PRGn(np.linspace(0, 0.3, len(generations)))
    crossover_colours = plt.cm.PRGn_r(np.linspace(0, 0.3, len(generations)))
    for structure in generations[0].populace:
        ax.scatter(structure['concentration'][0], structure['formation_enthalpy_per_atom'],
                   c='r', s=50, zorder=100000000, lw=2, edgecolors='k')
    for idx, generation in enumerate(generations[1:]):
        for structure in generation.populace:
            if 'mutations' not in structure:
                colour = mutant_colours[idx]
            else:
                colour = mutant_colours[idx] if 'crossover' not in structure['mutations'] else crossover_colours[idx]
            ax.scatter(structure['concentration'][0], structure['formation_enthalpy_per_atom'],
                       c=colour, s=35, zorder=1000, lw=0)
    colours = ['red', mutant_colours[0], crossover_colours[0]]
    labels = ['initial populace', 'mutant', 'crossover']
    for ind, colour in enumerate(colours):
        if ind == 0:
            ax2.scatter(-0.5, 10, c=colour, s=50, zorder=1000, lw=2, edgecolors='k', label=labels[ind])
        else:
            ax2.scatter(-0.5, 10, c=colour, s=35, zorder=1000, lw=0, label=labels[ind])
    ax2.legend(loc=9, frameon=True)
    sns.despine(ax=ax2, left=False, bottom=False)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(ax.get_ylim())
    # ax2.set_xticks([])
    ax2.set_yticks([])
    if save:
        plt.savefig(generations[0].run_hash + '_hullplot.pdf')
