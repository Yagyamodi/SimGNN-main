# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:25:45 2021

@author: 91876
"""

import networkx as nx

FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
for n, nbrs in FG.adj.items():
   for nbr, eattr in nbrs.items():
       wt = eattr['weight']
       if wt < 0.5: print(f"({n}, {nbr}, {wt:.3})")

for (u, v, wt) in FG.edges.data('weight'):
    if wt < 0.5:
        print(f"({u}, {v}, {wt:.3})")
        
g2 = FG.copy()
g2.add_weighted_edges_from([(2,3,0.75)])

from ged4py.algorithm import graph_edit_dist
print(graph_edit_dist.compare(FG,g2))