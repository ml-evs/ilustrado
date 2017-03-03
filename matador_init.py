#!/usr/bin/env python
from matador.query import DBQuery
from matador.hull import QueryConvexHull
from matador.export import doc2cell, doc2param
from ilustrado.ilustrado import ArtificialSelector

# prepare best structures from hull as gene pool
query = DBQuery(composition=['LiZn'], db=['LiPX'], subcmd='hull', biggest=True)
hull = QueryConvexHull(query,
                       subcmd='hull', no_plot=True,
                       summary=True, hull_cutoff=5e-2)
# lay out relaxation params
doc2cell(query.cursor[1], 'ga_test')
query.cursor[1]['geom_max_iter'] = 10
doc2param(query.cursor[1], 'ga_test')

# remove chempots
ArtificialSelector(gene_pool=hull.hull_cursor[1:-1],
                   seed='ga_test',
                   hull=hull,
                   debug=False,
                   # nnodes=4,
                   nodes=['node1', 'node2', 'node3', 'node4'],
                   ncores=16,
                   num_generations=10, population=50, num_survivors=10)
