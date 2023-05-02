import numpy as np
import networkx as nx
from mpi4py import MPI


class GraphConstruct:

    def __init__(self, rank, size, comm, graph, weight_type, p=0.5, num_c=None):

        # Initialize MPI variables
        self.rank = rank  # index of worker
        self.size = size  # total number of workers
        self.comm = comm

        # Create graph from string input or return custom inputted graph
        self.graph = self.selectGraph(graph, p, num_c)

        # if running in parallel
        if self.size > 1:
            # Determine each node's neighbors and the weights for each node in the Graph
            self.neighbor_list = self.getNeighbors(rank)
            self.neighbor_weights = self.getWeights(weight_type)
        else:
            self.neighbor_list = []
            self.neighbor_weights = np.zeros(1)

    def selectGraph(self, graph, p, num_c):

        if isinstance(graph, list):
            return graph
        else:
            g = []
            if graph == 'fully-connected':
                fc_graph = nx.complete_graph(self.size)
                g = fc_graph.edges

            elif graph == 'ring':
                ring_graph = nx.cycle_graph(self.size)
                g = ring_graph.edges

            elif graph == 'clique-ring':
                per_c = int(self.size/num_c)
                rem = self.size % num_c
                for i in range(num_c):
                    if i != num_c-1:
                        fc_graph = nx.complete_graph(per_c)
                        fc_graph = nx.convert_node_labels_to_integers(fc_graph, i*per_c)
                        g += fc_graph.edges
                        g.append((i*per_c + per_c-1, i*per_c + per_c))
                    else:
                        fc_graph = nx.complete_graph(per_c + rem)
                        fc_graph = nx.convert_node_labels_to_integers(fc_graph, i*per_c)
                        g += fc_graph.edges
                        if num_c > 2:
                            g.append((self.size-1, 0))

            elif graph == 'erdos-renyi':
                if self.rank == 0:
                    while True:
                        erdos_graph = nx.erdos_renyi_graph(self.size, p)
                        if nx.is_connected(erdos_graph):
                            g = erdos_graph.edges
                            num_edges = len(g)*np.ones(1, dtype=np.int)
                            print('Generated Erdos-Renyi Graph Edges:')
                            print(g)
                            break
                else:
                    num_edges = np.zeros(1, dtype=np.int)
                self.comm.Bcast(num_edges, root=0)
                num_edges = num_edges[0]
                if self.rank != 0:
                    data = np.empty((num_edges, 2), dtype=np.int)
                else:
                    data = np.array(g, dtype=np.int)
                self.comm.Bcast(data, root=0)
                if self.rank != 0:
                    for i in range(num_edges):
                        g.append((data[i][0], data[i][1]))
            return g

    def getWeights(self, weight_type=None):

        # uniform CIS assumed for this version
        # need to implement a non-uniform version...
        if weight_type == 'swift':

            degree = len(self.neighbor_list)
            requests = [MPI.REQUEST_NULL for _ in range(degree)]
            # send degree info to all neighbors
            send_buff = np.zeros(2)
            send_buff[0] = degree
            for i, node in enumerate(self.neighbor_list):
                requests[i] = self.comm.Isend(send_buff, dest=node, tag=self.rank + 100*self.size)

            # receive neighboring degrees (blocking)
            neighbor_degrees = np.empty(degree)
            recv_buff = np.empty(2)
            for idx, node in enumerate(self.neighbor_list):
                self.comm.Recv(recv_buff, source=node, tag=node + 100*self.size)
                neighbor_degrees[idx] = recv_buff[0]

            sort_idx = np.argsort(-neighbor_degrees)
            sorted_nd = neighbor_degrees[sort_idx]
            sorted_nn = np.asarray(self.neighbor_list)[sort_idx]

            # clear memory
            for j in range(degree):
                if requests[j].Test():
                    requests[j].Wait()

            # receive and then send weights to neighbors
            # if you are the largest degree node in the graph, you get the ball rolling
            if degree >= np.max(neighbor_degrees):
                weights = (1 / (degree + 1)) * np.ones(degree)
                send_buff = np.zeros(2)
                send_buff[0] = 1/(degree + 1)
                # start setting weights and sending them
                for i, node in enumerate(sorted_nn):
                    requests[i] = self.comm.Isend(send_buff, dest=node, tag=self.rank + 200*self.size)
            else:
                weights = np.zeros(degree)
                recv_buff = np.empty(2)
                # receive until you are the largest left and then send
                while degree < sorted_nd[0]:
                    index = sort_idx[0]
                    self.comm.Recv(recv_buff, source=sorted_nn[0], tag=sorted_nn[0] + 200*self.size)
                    weights[index] = recv_buff[0]
                    sort_idx = sort_idx[1:]
                    sorted_nn = sorted_nn[1:]
                    sorted_nd = sorted_nd[1:]
                    if sorted_nd.size == 0:
                        break

                if sorted_nd.size > 0:

                    weight_sum = np.sum(weights)
                    uniform_weight = (1 - weight_sum) / (sorted_nd.size + 1)
                    send_buff = np.zeros(2)
                    send_buff[0] = uniform_weight
                    recv_buff = np.empty(2)

                    # decide on weights if neighbors have the same degree
                    if degree == sorted_nd[0]:
                        same_degree_neighbors = sorted_nn[sorted_nd == degree]
                        comp_weights = np.zeros(len(same_degree_neighbors))
                        for i, node in enumerate(same_degree_neighbors):
                            requests[i] = self.comm.Isend(send_buff, dest=node, tag=self.rank + 200 * self.size)
                        for j, node in enumerate(same_degree_neighbors):
                            self.comm.Recv(recv_buff, source=node, tag=node + 200 * self.size)
                            comp_weights[j] = recv_buff[0]
                        # if you share same degree as neighboring node, choose the weighting that's smaller to share
                        # in order to assure that this process works every time
                        if uniform_weight > np.min(comp_weights):
                            uniform_weight = np.min(comp_weights)

                    weights[weights == 0] = uniform_weight
                    send_buff = np.zeros(2)
                    send_buff[0] = uniform_weight
                    for node in sorted_nn[sorted_nd != degree]:
                        self.comm.Send(send_buff, dest=node, tag=self.rank + 200 * self.size)

            self.comm.Barrier()
            # clear memory
            for j in range(degree):
                if requests[j].Test():
                    requests[j].Wait()

        elif weight_type == 'uniform-symmetric':
            num_neighbors = len(self.neighbor_list)
            weights = (1 / self.size) * np.ones(num_neighbors)

        # Neighborhood uniform weights by default
        else:
            num_neighbors = len(self.neighbor_list)
            weights = (1/(num_neighbors+1)) * np.ones(num_neighbors)

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
    