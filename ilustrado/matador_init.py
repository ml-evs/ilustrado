#!/usr/bin/env python
from matador.query import DBQuery
from matador.hull import QueryConvexHull
from ilustrado import ArtificialSelector

query = DBQuery(composition=['LiZn'], db=['LiPX'], subcmd='hull', biggest=True)
hull = QueryConvexHull(query,
                       subcmd='hull', no_plot=True,
                       summary=True, hull_cutoff=5e-2)

ArtificialSelector(gene_pool=hull.hull_cursor, debug=False)
