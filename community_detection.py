from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np

# good resource
# http://vw.indiana.edu/netsci06/conf-slides/conf-mon/netsci-talk-mark-newman.pdf

# TODO
# done: made get_B, and make adjacency matrix faster via taking advantage of symmetry
# seems like get_B takes a long time

# and modularity() takes a long time
# paper has some points on computing delta modularity more quickly

def get_adj_dict(which_set):
    """
    returns a symmettric adjacency dictionary
    where keys are nodes and value is a list of nodes which the key_node has an edge to
    i.e. for patents, keys are patent numbers and value is a list of patents which cite it

    """
    # keys are patent numbers is a list of patents which cite it
    if which_set == 'patents':
        # root patent is 4723129 is k[55]
        return np.load('alex_adj.p', mmap_mode='r')
    elif which_set == 'wikipedia':
        # from the wikipedia on modularity
        adj_dict = {
               1:  [2,3,10],
               2:  [1,3],
               3:  [1,2],
               4:  [5,6,10],
               5:  [4,6],
               6:  [4,5],
               7:  [8,9,10],
               8:  [7,9],
               9:  [7,8],
               10: [1,4,7],
               }
        return adj_dict
    elif which_set == 'tester':
        adj_dict = {
                1: [2,3,4],
                2: [1,3],
                3: [1,2],
                4: [1,5,6],
                5: [4,6],
                6: [4,5]
                }
        return adj_dict
    else:
        raise RuntimeError('%s is not a set we have' % which_set)

def get_adjaceny_matrix(adj_dict):
    flattened_values= [x for xs in list(adj_dict.values()) for x in xs]
    nodes = sorted(list(set(list(adj_dict.keys())+flattened_values)))
    A = np.zeros((len(nodes), len(nodes)))
    for i, _ in enumerate(A):
        for j in range(i+1):
            patent1 = nodes[i]
            patent2 = nodes[j]
            flag = False
            if patent1 in adj_dict:
                if patent2 in adj_dict[patent1]:
                    flag = True
            if patent2 in adj_dict:
                if patent1 in adj_dict[patent2]:
                    flag = True
            A[i,j] = A[j,i] = 1 if flag else 0
    return A

def get_initial_S(which_set, num_nodes):
    """
    make S, a num_nodes by num_communities matrix
    where S[i,j] = 1 if patent i belongs to community j else 0
    """
    S = None
    if which_set == 'generic':
        S = np.zeros((num_nodes, num_nodes))
        np.fill_diagonal(S, 1)
    elif which_set == 'wikipedia':
        S = np.zeros((num_nodes, 3))
        a = [1,0,0]
        b = [0,1,0]
        c = [0,0,1]
        for i, _ in enumerate(S):
            if i in [0,1,2]:
                S[i] = a
            elif i in [3,4,5]:
                S[i] = b
            else:
                S[i] = c
    elif which_set == 'tester':
        S = np.zeros((6,2))
        S[1,0] = 1
        S[2,0] = 1
        S[3,1] = 1
        S[4,1] = 1
        S[5,1] = 1
    else:
        raise RuntimeError('no such condition %s' % which_set)
    return S

def get_B(A):
    """
    make B, the modularity matrix, which has dimensions num_nodes by num_nodes.
    B[i,j] = A[i,j] - ((k_i * k_j) / (2m))
    where m is the number of edges (2m = num_stubs)
    and where A is adjacency matrix and k_i is degree.
    """

    def edge_deg(node_idx):
        return np.sum(A[node_idx])

    num_stubs = get_num_stubs(A)
    num_nodes = A.shape[0]
    B = np.zeros((num_nodes, num_nodes))
    for i, row in enumerate(B):
        for j in range(i+1):
            B[i,j] = B[j,i] = A[i,j] - float((edge_deg(i) * edge_deg(j)) / (num_stubs))
    return B

def modularity(A, S):
    """
    computes modularity, Q, where Q = 1/2m * Tr(S^t B S)
    """
    B = get_B(A)
    num_stubs = get_num_stubs(A)
    return float(1 / num_stubs) * np.trace(np.dot(np.transpose(S), np.dot(B,S)))

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
    num_stubs = A.shape[0]
    counter = 0
    wasChangedInFunction = False
    wasChangedInLoop = True
    while wasChangedInLoop:
        wasChangedInLoop = False
        print('phase1_counter: %d' % counter)
        counter+=1
        # loop over each node
        # this for loop takes fooooorever
        for i, S_row in enumerate(S):
            cur_community = best_community = np.nonzero(S_row)[0][0]

            # remove node from its former community
            S_row[cur_community] = 0

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
    print('starting phase2')
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
    # formula:
    # sum over all nodes j in community
    # (1/num_stubs) * (2 * (A_ij - (k_i * k_j) / num_stubs) + (A_ii - (k_i*k_i)/num_stubs))
    #
    # seems to be off in the third decimal point
    # not sure why
    #
    # returns the value of adding node_i to community
    # simply multiply the value by negative 1 to get the value
    # of removing node i from the community

    k_dict = {}
    def k(node_idx):
        if node_idx in k_dict:
            return k_dict[node_idx]
        else:
            val =  np.sum(A[node_idx])
            k_dict[node_idx] = val
            return val

    cum_sum = 0
    for j in np.nonzero(S[:,community])[0]:
        cum_sum += (A[node_i,j] - ((k(node_i) * k(j)) / num_stubs))
    cum_sum = 2 * cum_sum
    cum_sum += A[node_i, node_i] - ((k(node_i)**2) / num_stubs)
    cum_sum = cum_sum / num_stubs
    return cum_sum

def go(A,S):
    counter = 0
    node_comm_associations = [[i] for i in range(A.shape[0])]
    while True:
        print ('iteration %d' % counter)
        counter+=1
        S, wasChanged = phase1(A,S)
        if wasChanged == False:
            break
        A, S, node_comm_associations = phase2(A, S, node_comm_associations)
        break
    return A, S, node_comm_associations

if __name__ == '__main__':
    print('main')
    adj_dict = get_adj_dict('wikipedia')
    A = get_adjaceny_matrix(adj_dict)
    num_nodes = A.shape[0]
    S = get_initial_S('generic', num_nodes)
    # s,b = phase1(A,S)
    a,s,n = go(A,S)

    # f = delta_modularity(0,0,A,S)
    # g = modularity(A, S, B)


    # x,y = phase1(A, S)
    # print(x)
    # print(y)

    #X,Y,Z = go(A,S)
    #print(X)
    #print(Y)
    #print(Z)





