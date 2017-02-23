#!/usr/bin/env python
from matador.query import DBQuery
from immacolata import ArtificialSelector

query = DBQuery(id='apostle underwear', db=['LiPX'], top=1)

ArtificialSelector(gene_pool=query.cursor)
