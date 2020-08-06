def testing_with_equation_function(number_of_tasks,features_indices,number_of_steps,F,Q):

    # print("\n\nExecuting 'testing.py' ...")


    #%% Import Statements:

    from observations_to_state_release_times import observations_to_state_release_times_function
    from observations_to_state_weights import observations_to_state_weights_function
    from observations_to_state_lengths import observations_to_state_lengths_function
    from observations_to_state_q_matrix import observations_to_state_q_matrix_function

    import numpy


    #%% Definitions of Variables Based on User-Specified Inputs:

    r_N_index = features_indices[0]
    w_N_index = features_indices[1]
    l_N_index = features_indices[2]

    r_N_steps = number_of_steps[0]
    w_N_steps = number_of_steps[1]
    l_N_steps = number_of_steps[2]

    r_N_testing = F[:,(r_N_index-1)*number_of_tasks:(r_N_index*number_of_tasks)]
    w_N_testing = F[:,(w_N_index-1)*number_of_tasks:(w_N_index*number_of_tasks)]
    l_N_testing = F[:,(l_N_index-1)*number_of_tasks:(l_N_index*number_of_tasks)]


    #%% Testing:

    results = numpy.zeros([len(F)])

    for i in range(len(F)):

        r_continuous = r_N_testing[i,:]
        w_continuous = w_N_testing[i,:]
        l_continuous = l_N_testing[i,:]

        r_state = observations_to_state_release_times_function(number_of_tasks,r_continuous,r_N_steps)
        w_state = observations_to_state_weights_function(number_of_tasks,w_continuous,w_N_steps)
        l_state = observations_to_state_lengths_function(number_of_tasks,l_continuous,l_N_steps)
        
        Q_state = observations_to_state_q_matrix_function(number_of_tasks,r_N_steps,w_N_steps,l_N_steps,r_state,w_state,l_state)

        action = numpy.argmax(Q[Q_state][:])

        results[i] = action

    # print("\nResults = \n\n{}".format(results))
    # print("\n\tShape of Results = {}".format(numpy.shape(results)))

    # print("\n\tTESTING FINISHED")


    #%% Export and Return Statements:

    return results


    #%% Documentation:

    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    # https://numpy.org/doc/stable/reference/generated/numpy.shape.html