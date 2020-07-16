def observations_to_state_function(r_continuous,w_continuous,l_continuous,r_number_of_states,w_number_of_states,l_number_of_states):

    # print("\n\nExecuting 'observations_to_state.py' ...")


    ##### Import Statements #####

    import numpy


    ##### Mapping Observations to State #####

    r_N_low = 0
    r_N_high = 1
    r_N_dx = (r_N_high-r_N_low)/r_number_of_states

    w_N_low = 0
    w_N_high = 10
    w_N_dx = (w_N_high-w_N_low)/w_number_of_states

    l_N_low = 0
    l_N_high = 1
    l_N_dx = (l_N_high-l_N_low)/l_number_of_states

    r_discrete = numpy.zeros([len(r_continuous)])
    w_discrete = numpy.zeros([len(w_continuous)])
    l_discrete = numpy.zeros([len(l_continuous)])

    for i in range(len(r_discrete)):
        r_discrete[i] = int((r_continuous[i]-r_N_low)/r_N_dx)
        w_discrete[i] = int((w_continuous[i]-w_N_low)/w_N_dx)
        l_discrete[i] = int((l_continuous[i]-l_N_low)/l_N_dx)

    r_number_of_elements = r_number_of_states**3
    w_number_of_elements = w_number_of_states**3
    l_number_of_elements = l_number_of_states**3

    r_state_matrix = numpy.array(numpy.arange(0,r_number_of_elements)).reshape(r_number_of_states,r_number_of_states,r_number_of_states)
    w_state_matrix = numpy.array(numpy.arange(0,w_number_of_elements)).reshape(w_number_of_states,w_number_of_states,w_number_of_states)
    l_state_matrix = numpy.array(numpy.arange(0,l_number_of_elements)).reshape(l_number_of_states,l_number_of_states,l_number_of_states)

    r_state = r_state_matrix[int(r_discrete[0]),int(r_discrete[1]),int(r_discrete[2])]
    w_state = w_state_matrix[int(w_discrete[0]),int(w_discrete[1]),int(w_discrete[2])]
    l_state = l_state_matrix[int(l_discrete[0]),int(l_discrete[1]),int(l_discrete[2])]


    ##### Export and Return Statements #####

    return r_state,w_state,l_state


    ##### Documentation #####

    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # https://numpy.org/doc/stable/reference/generated/numpy.array.html
    # https://numpy.org/doc/stable/reference/generated/numpy.arange.html