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
# query.cursor[1]['geom_max_iter'] = 100
# print(query.cursor[1]['geom_max_iter'])
# doc2param(query.cursor[1], 'ga_test')
# print(query.cursor[1]['geom_max_iter'])

ArtificialSelector(gene_pool=hull.hull_cursor, seed='ga_test', hull=hull, debug=False)
