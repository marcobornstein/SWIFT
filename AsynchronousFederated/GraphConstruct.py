from mpi4py import MPI
import numpy as np


class GraphConstruct:

    def __init__(self, graph, rank, size):

        # Initialize MPI variables
        self.rank = rank  # index of worker
        self.size = size  # totoal number of workers
        self.comm = MPI.COMM_WORLD

        self.graph = self.selectGraph(graph)

        # Determine each node's neighbors and the weights for each node in the Graph
        self.neighbor_list = self.getNeighbors(rank)
        self.neighbor_weights = self.getWeights()

    def selectGraph(self, graph):

        if isinstance(graph, list):
            return graph
        else:
            g = []
            if graph == 'fully-connected':
                for i in range(self.size):
                    for j in range(i+1, self.size):
                        g.append((i, j))
            elif graph == 'ring':
                for i in range(self.size):
                    if i != self.size - 1:
                        g.append((i, i+1))
                    else:
                        g.append((i, 0))
            return g

    def getWeights(self, weight_type=None):
        
        if weight_type == 'learnable':
            # NEED TO IMPLEMENT HERE
            weights = np.ones(self.size)
            
        elif weight_type == 'matcha':
            # NEED TO IMPLEMENT HERE
            weights = np.ones(self.size)
            
        else:
            num_neighbors = len(self.neighbor_list)
            weights = (1/self.size) * np.ones(num_neighbors)
            
        return weights

    def getNeighbors(self, rank):
        
        neighbors = [[] for _ in range(self.size)]
        for edge in self.graph:
            node1, node2 = edge[0], edge[1]
            if node1 == node2:
                print("Invalid input graph! Circle! ("+str(node1) +", "+ str(node2)+")")
                exit()
            neighbors[node1].append(node2)
            neighbors[node2].append(node1)
            
        return neighbors[rank]
    