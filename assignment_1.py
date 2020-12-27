import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def one():
    def get_len_reciprocated_edges(G):
        c = 0
        for e in G.edges:
            if (e[0] != e[1]) and (e[1], e[0]) in G.edges:
                c += 1
        return int(c / 2)

    def get_len_out_degree(G, val_out_degree):
        out_degree_view = G.out_degree()
        c = 0
        for elem in out_degree_view:
            if elem[1] == val_out_degree:
                c += 1
        return c

    def get_len_in_degree(G, val_in_degree):
        in_degree_view = G.in_degree()
        c = 0
        for elem in in_degree_view:
            if elem[1] == val_in_degree:
                c += 1
        return c

    def get_len_out_degree_gt_10(G):
        out_degree_view = G.out_degree()
        c = 0
        for elem in out_degree_view:
            if elem[1] > 10:
                c += 1
        return c

    def get_len_in_degree_lt_10(G):
        in_degree_view = G.in_degree()
        c = 0
        for elem in in_degree_view:
            if elem[1] < 10:
                c += 1
        return c

    # file_name = "test_1.txt"
    file_name = "Wiki-Vote.txt"
    dG = nx.read_edgelist(file_name, create_using=nx.DiGraph)
    uG = dG.to_undirected()
    no_edges = dG.number_of_edges()
    no_edges_uG = uG.number_of_edges()
    a1 = dG.number_of_nodes()
    a2 = len(list(nx.selfloop_edges(dG)))
    a3 = no_edges - a2
    a4 = no_edges_uG - a2
    a5 = get_len_reciprocated_edges(dG)
    a6 = get_len_out_degree(dG, 0)
    a7 = get_len_in_degree(dG, 0)
    a8 = get_len_out_degree_gt_10(dG)
    a9 = get_len_in_degree_lt_10(dG)

    print("The number of nodes in the network : ", a1)
    print("The number of nodes with a self-edge (self-loop) : ", a2)
    print("The number of directed edges in the network : ", a3)
    print("The number of undirected edges in the network : ", a4)
    print("The number of reciprocated edges in the network : ", a5)
    print("The number of nodes of zero out-degree : ", a6)
    print("The number of nodes of zero in-degree : ", a7)
    print("The number of nodes with more than 10 outgoing edges : ", a8)
    print("The number of nodes with fewer than 10 incoming edges : ", a9)


def two():
    file_name = "Wiki-Vote.txt"
    dG = nx.read_edgelist(file_name, create_using=nx.DiGraph)
    list_od = dG.out_degree()
    list_sod = sorted(list_od, key=lambda tup: tup[1])
    min_x = list_sod[0][1]
    max_x = list_sod[-1][1]
    dict_od = {}
    for elem in list_sod:
        if elem[1] not in dict_od.keys():
            dict_od[elem[1]] = 1
        else:
            dict_od[elem[1]] += 1
    list_x = []
    list_y = []
    for k, v in dict_od.items():
        list_x.append(k)
        list_y.append(v)
    fig1 = plt.figure(constrained_layout=True)
    spec1 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig1)
    ax1 = fig1.add_subplot(spec1[0, 0])
    fig1.suptitle('Plot : Out Degree Distribution (scale: log-log)')
    # ax1.plot(list_x, list_y, color='blue', marker='.')
    ax1.loglog(list_x, list_y, color='blue', marker='.')
    ax1.set_xlabel('Out Degree')
    ax1.set_ylabel('Count')


def three():
    file_name = "stackoverflow-java.txt"
    dG_stack = nx.read_edgelist(file_name, create_using=nx.DiGraph)
    no_wcc_stack = len(list(nx.weakly_connected_components(dG_stack)))
    largest_wcc_stack = dG_stack.subgraph(sorted(nx.weakly_connected_components(dG_stack), key=len, reverse=True)[0])
    no_nodes_lwccs = largest_wcc_stack.number_of_nodes()
    no_edges_lwccs = largest_wcc_stack.number_of_edges()
    print("1. The number of weakly connected components in the network : ", no_wcc_stack)
    print("2a. The number of nodes in the largest weakly connected component : ", no_nodes_lwccs)
    print("2b. The number of edges in the largest weakly connected component : ", no_edges_lwccs)


def four():
    import random
    class CustomGnm:
        def __init__(self, n, m, directed=False):
            self.n = n
            self.m = m
            self.directed = directed
            self.nodes = self.build_nodes()
            self.edges = self.build_edges()
            self.graph = self.build_graph()

        def build_nodes(self):
            return [i for i in range(self.n)]

        def build_edges(self):
            list_edges = []
            list_possible_edges = []
            if not self.directed:
                i = 0
                for i in range(self.n):
                    a = self.nodes[i]
                    for b in self.nodes[i+1:]:
                        list_possible_edges.append((a,b))
            len_possible_edges = (len(list_possible_edges))
            count_edge = 0
            while (count_edge < self.m):
                i = random.randint(0,len_possible_edges-1)
                possible_edge = list_possible_edges[i]
                if possible_edge not in list_edges:
                    list_edges.append(possible_edge)
                    count_edge +=1
            return list_edges

        def build_graph(self):
            G = nx.Graph()
            G.add_nodes_from(self.nodes)
            G.add_edges_from(self.edges)
            return G.to_undirected()

    class CustomSmallWorldRandomNetwork:
        def __init__(self, n, m=0, directed=False, no_random_edge=0):
            self.no_random_edge = no_random_edge
            self.n = n
            self.directed = directed
            self.nodes = self.build_nodes()
            self.edges = self.build_edges()
            self.m = len(self.edges)
            if self.m != m:
                raise Exception("Error on Graph Formulation!!!")
            self.graph = self.build_graph()

        def build_nodes(self):
            return [i for i in range(self.n)]

        def build_edges(self):
            list_edges = []
            if not self.directed:
                i = 0
                for i in range(self.n):
                    a = self.nodes[i]
                    next_1_node = (i+1)%self.n
                    prev_1_node = (self.n + i-1)%self.n
                    next_2_node = (i+2)%self.n
                    prev_2_node = (self.n + i-2)%self.n
                    if (a,next_1_node) not in list_edges:
                        list_edges.append((a,next_1_node))
                    if (prev_1_node,a) not in list_edges:
                        list_edges.append((prev_1_node,a))
                    if (a,next_2_node) not in list_edges:
                        list_edges.append((a,next_2_node))
                    if (prev_2_node,a) not in list_edges:
                        list_edges.append((prev_2_node,a))
            count_edge = 0
            while (count_edge < self.no_random_edge):
                a = random.randint(0,self.n-1)
                b = random.randint(0,self.n-1)
                if a != b:
                    if (a,b) not in list_edges and (b,a) not in list_edges:
                        list_edges.append((a,b))
                        count_edge +=1
            return list_edges

        def build_graph(self):
            G = nx.Graph()
            G.add_nodes_from(self.nodes)
            G.add_edges_from(self.edges)
            return G.to_undirected()

    def report_property(G):
        # degree_distribution_G
        # clustering_coefficient_G
        # diameter_G
        print("(#nodes, #edges) : ", (len(G.nodes),len(G.edges)))
        # print(G.edges)

    cgnm = CustomGnm(5242,14484)
    cswrn = CustomSmallWorldRandomNetwork(5242,14484,no_random_edge=4000)
    ngnm = nx.gnm_random_graph(5242,14484)
    # cgnm = CustomGnm(6,14)
    # cswrn = CustomSmallWorldRandomNetwork(6,14,no_random_edge=2)
    # ngnm = nx.gnm_random_graph(6,14)
    report_property(cgnm.graph)
    report_property(cswrn.graph)
    report_property(ngnm)

def five():
    pass

if __name__=='__main__':
    # one()
    # two()
    # three()
    four()
    five()