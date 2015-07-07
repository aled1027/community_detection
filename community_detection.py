from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np

# good resource
# http://vw.indiana.edu/netsci06/conf-slides/conf-mon/netsci-talk-mark-newman.pdf

def get_adj_dict():
    """
    returns an adjacency matrix where keys are nodes and value is a list
    of nodes which the key_node has an edge to
    i.e. for patents, keys are patent numbers and value is a list of
    patents which cite it
    """
    return np.load('alex_adj.p', mmap_mode='r')

def get_adjaceny_matrix(adj_dict):
    """
    converts an adjacency dictionary into a symmetri
    """
    flattened_values= [x for xs in list(adj_dict.values()) for x in xs]
    nodes = sorted(list(set(list(adj_dict.keys())+flattened_values)))
    A = np.zeros((len(nodes), len(nodes)))
    for i, _ in enumerate(A):
        for j in range(i+1):
            node1 = nodes[i]
            node2 = nodes[j]
            flag = False
            if node1 in adj_dict and node2 in adj_dict[node1]:
                    flag = True
            elif node2 in adj_dict and node1 in adj_dict[node2]:
                    flag = True
            A[i,j] = A[j,i] = 1 if flag else 0
    return A

def convert_communities_to_patents(adj_dict, communities):
    flattened_values = [x for xs in list(adj_dict.values()) for x in xs]
    nodes = sorted(list(set(list(adj_dict.keys()) + flattened_values)))
    def convert_to_patent(idx):
        return nodes[idx]
    r = [list(map(convert_to_patent, community)) for community in communities]

    return r
c = convert_communities_to_patents



def get_initial_S(num_nodes):
    """
    initial S has each node its own community
    """
    S = np.zeros((num_nodes, num_nodes))
    np.fill_diagonal(S, 1)
    return S

def get_num_stubs(A):
    """
    A is an adjacency matrix.
    An edge is a composition of two stubs.
    Returns number of stubs, aka 2*num_edges.
    """
    return np.sum(A)

def phase1(A, S):
    """
    phase1 takes the graph A and S and returns a better S
    phase2 then takes S and squashes communities, returning a new S and A

    S[i,c] = 1 if node i belongs to community c else 0
    """
    # loop over nodes, finding a local max of Q
    num_stubs = get_num_stubs(A)
    counter = 0
    wasChangedInFunction = False
    wasChangedInLoop = True
    while wasChangedInLoop:
        wasChangedInLoop = False
        print('    phase1 counter: %d' % counter)
        counter+=1
        # loop over each node
        # this for loop takes fooooorever
        for i, S_row in enumerate(S):
            cur_community = best_community = np.nonzero(S_row)[0][0]

            # remove node from its former community
            S[i, cur_community] = 0
            # find best delta Q for all other communities
            best_delta_Q = -10.0
            for j, _ in enumerate(S_row):
                delta_Q = delta_modularity(i, j, num_stubs, A, S)
                if delta_Q > best_delta_Q:
                    best_delta_Q = delta_Q
                    best_community = j
            if cur_community != best_community:
                wasChangedInLoop= True
                wasChangedInFunction= True
            S[i, best_community] = 1

    # remove columns that all zeros via a mask
    # this removes irrelevant communities
    S = np.transpose(S)
    S = np.transpose(S[(S!=0).any(axis=1)])
    return S, wasChangedInFunction

def phase2(A, S, node_comm_associations):
    """
    squash communities
    """
    print('    starting phase2')
    # So S = num_nodes by num_communities
    # so we are going to have
    num_communities = S.shape[1]
    new_A = np.zeros((num_communities, num_communities))

    # fill new_A
    for i, row in enumerate(new_A):
        for j, _ in enumerate(row):
            # get set of nodes in community i and
            comm_i_nodes = np.nonzero(S[:,i])[0]
            comm_j_nodes = np.nonzero(S[:,j])[0]

            # get number of edge intersections
            edge_sum = 0
            for comm_i_node in comm_i_nodes:
                for comm_j_node in comm_j_nodes:
                    edge_sum += A[comm_i_node, comm_j_node]
            new_A[i,j] = edge_sum
        new_A[i,i] = 0.5 * new_A[i,i]

    # update node_comm_associations
    new_node_comm_associations = []

    # loop over columns
    S = np.transpose(S)
    for row in S:
        nodes = np.nonzero(row)[0]
        # combine old nodes of node_comm_associations
        temp_list = [x for y in nodes for x in node_comm_associations[y]]
        new_node_comm_associations.append(temp_list)

    # also need a list of all original nodes associated with each community
    new_S = np.zeros((num_communities, num_communities))
    for i, _ in enumerate(new_S):
        new_S[i,i] = 1
    return new_A, new_S, new_node_comm_associations


def delta_modularity(node_i, community, num_stubs, A, S):
    """
    formula:
    sum over all nodes j in community
    (1/num_stubs) * (2 * (A_ij - (k_i * k_j) / num_stubs) + (A_ii - (k_i*k_i)/num_stubs))

    returns the value of adding node_i to community
    simply multiply the value by negative 1 to get the value
    of removing node i from the community
    """
    k_dict = {}
    def k(node_idx):
        """
        returns k_i, the number of stubs that a node has, aka its outdegree
        """
        if node_idx in k_dict:
            return k_dict[node_idx]
        else:
            val =  np.sum(A[node_idx])
            k_dict[node_idx] = val
            return val

    cum_sum = 0
    # loop over members of community
    for j in np.nonzero(S[:,community])[0]:
        cum_sum += (A[node_i,j] - ((k(node_i) * k(j)) / num_stubs))
    cum_sum = 2 * cum_sum

    # add in value for node_i
    cum_sum += A[node_i, node_i] - ((k(node_i)**2) / num_stubs)

    cum_sum = cum_sum / num_stubs
    return cum_sum

def run():
    """
    Main running function for algorithm.
    Runs phase1 and then phase2. rinse and repeat while necessary
    """
    adj_dict = get_adj_dict()
    A = get_adjaceny_matrix(adj_dict)
    num_nodes = A.shape[0]
    S = get_initial_S(num_nodes)


    counter = 0
    # node_comm_associations is a list containing
    # jwhich nodes are in which communities
    # e.g say node 1 and 2 are in a community and node 3 in a diff. community
    # then node_comm_associations = [[1,2], [3]]
    node_comm_associations = [[i] for i in range(A.shape[0])]
    while True:
        print ('go counter: %d' % counter)
        counter+=1
        S, wasChanged = phase1(A,S)
        if wasChanged == False:
            break
        A, S, node_comm_associations = phase2(A, S, node_comm_associations)

    communities = convert_communities_to_patents(adj_dict, node_comm_associations)

    for c in communities:
        print(c)
    return communities

if __name__ == '__main__':
    print('running')
    run()

