#!/usr/bin/env python
from matador.query import DBQuery
from ilustrado import ArtificialSelector

query = DBQuery(id='apostle underwear', db=['LiPX'], top=1)

ArtificialSelector(gene_pool=query.cursor, debug=False)
