def testing_function(number_of_tasks,F,Q,number_of_states):

    print("\n\nExecuting 'testing.py' ...")
    

    ##### Import Statements #####

    from observations_to_state import observations_to_state_function

    import numpy


    ##### Testing #####

    results = numpy.zeros([len(F)])

    for i in range(len(F)):

        r_continuous = F[i,0:1*number_of_tasks]
        w_continuous = F[i,1*number_of_tasks:2*number_of_tasks]
        l_continuous = F[i,2*number_of_tasks:3*number_of_tasks]

        r_number_of_states = number_of_states[0]
        w_number_of_states = number_of_states[1]
        l_number_of_states = number_of_states[2]

        r_state,w_state,l_state = observations_to_state_function(r_continuous,w_continuous,l_continuous,r_number_of_states,w_number_of_states,l_number_of_states)

        action = numpy.argmax(Q[r_state][w_state][l_state][:])

        results[i] = action

    # print("\nResults = \n\n{}".format(results))
    print("\n\tShape of Results = {}".format(numpy.shape(results)))

    print("\n\tTESTING FINISHED")


    ##### Export and Return Statements #####

    return results


    ##### Documentation #####

    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    # https://numpy.org/doc/1.18/reference/generated/numpy.shape.html