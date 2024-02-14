# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:09:19 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGraphData, loaderGNN, fixEdgesAndClustersForBatch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# Simulate some data to make a loader from
n_samples = 10
n_nodes = 8
x, edge_index_list, clusters_list = simulateGraphData(
    n_samples = n_samples, 
    n_nodes = n_nodes,
    n_features = 5, 
    n_edges = 10,  
    n_pooling_layers = 1
)
y, _, _ = simulateGraphData(
    n_samples = n_samples, 
    n_nodes = n_nodes,
    n_features = 1,  
    n_edges = 0,
    n_pooling_layers = 0
)

# Organize the data
data = (x, y, edge_index_list, clusters_list)

# Make a default loader batch_size=1, shuffle=False
loader1 = loaderGNN(data)
batch1 = next(iter(loader1))
print("batch.x:", batch1.x.size())
print("batch.y:", batch1.y.size())
print("batch.batch:", batch1.batch.size())
#print(batch1.batch)

# Make a new loader batch_size=3, shuffle=False
loader2 = loaderGNN(data, batch_size=3)
batch2 = next(iter(loader2))
print("batch.x:", batch2.x.size())
print("batch.y:", batch2.y.size())
print("batch.batch:", batch2.batch.size())
#print(batch2.batch)


## Test one more thing with the edge_index part
edges = edge_index_list[0]
print(edges.size())
dataset = []
for i in range(data[0].size(0)):
    dataset.append(Data(edge_index=edges, x=data[0][i, :, :], y=data[1][i, :, :]))
loader3 = DataLoader(dataset, batch_size=3, shuffle=False)
batch3 = next(iter(loader3))
print("batch.edge_index:", batch3.edge_index.size())
print("batch.x:", batch3.x.size())
print("batch.y:", batch3.y.size())
print("batch.batch:", batch3.batch.size())
#print(batch3.batch)

## Need to make a function that fixes the edges to be like batch.edge_index
print(edge_index_list)
print(clusters_list)
edge_index_list_fixed, clusters_list_fixed = fixEdgesAndClustersForBatch(edge_index_list, clusters_list, batch2.batch)
print(edge_index_list_fixed)
print(clusters_list_fixed)
