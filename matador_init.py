#!/usr/bin/env python
from matador.query import DBQuery
from matador.hull import QueryConvexHull
from ilustrado.ilustrado import ArtificialSelector

# prepare best structures from hull as gene pool
query = DBQuery(composition=['LiZn'], db=['LiPX'], subcmd='hull', biggest=True)
hull = QueryConvexHull(query,
                       subcmd='hull', no_plot=True,
                       summary=True, hull_cutoff=1e-2)
# lay out relaxation params
# doc2cell(query.cursor[1], 'ga_test')

# remove chempots
ArtificialSelector(gene_pool=hull.hull_cursor[1:-1],
                   seed='ga_test',
                   hull=hull,
                   debug=True,
                   fitness_metric='hull',
                   nnodes=1,
                   nodes=['node21'],
                   ncores=20,
                   num_generations=5, population=2, num_survivors=2)
