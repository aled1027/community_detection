
def time_tests():
    def time_function(f, arg1=None, arg2=None, arg3=None):
        import time
        start = time.time()
        if arg1 == None:
            ret = f()
        elif arg2 == None:
            ret = f(arg1)
        elif arg3 == None:
            ret = f(arg1, arg2)
        else:
            ret = f(arg1, arg2, arg3)
        elapsed = time.time() - start
        return ret, elapsed

    print('running time tests')

    # SETUP
    adj_dict_which_set = 'patents'
    # adj_dict_which_set = 'wikipedia'
    adj_dict = get_adj_dict(adj_dict_which_set)

    # time get_adjacency_matrix
    A, time_elapsed = time_function(get_adjaceny_matrix, adj_dict)
    print ('get_adjaceny_matrix: %f' % time_elapsed)

    num_nodes = A.shape[0]
    S = get_initial_S('generic', num_nodes)

    # time get_B
    B, time_elapsed = time_function(get_B, A)
    print ('get_B: %f' % time_elapsed)

    #time modularity
    Q, time_elapsed = time_function(modularity, A, S, B)
    print ('modularity: %f' % time_elapsed)


