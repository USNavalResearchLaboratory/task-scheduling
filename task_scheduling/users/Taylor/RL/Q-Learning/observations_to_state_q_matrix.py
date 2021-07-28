def observations_to_state_q_matrix_function(number_of_tasks,r_N_steps,w_N_steps,l_N_steps,r_state,w_state,l_state):

    # print("\n\nExecuting 'observations_to_state_q_matrix.py' ...")


    #%% Import Statements:

    import numpy


    #%% Mapping Observations to State:

    r_number_of_elements = r_N_steps**number_of_tasks
    w_number_of_elements = w_N_steps**number_of_tasks
    # l_number_of_elements = l_N_steps**number_of_tasks
    
    # q_number_of_elements = r_number_of_elements*w_number_of_elements*l_number_of_elements
    # q_states = numpy.array(numpy.arange(0,q_number_of_elements)).reshape(r_number_of_elements,w_number_of_elements,l_number_of_elements)
    # q_state = q_states[int(r_state),int(w_state),int(l_state)]

    q_state = r_state+w_state*r_number_of_elements+l_state*r_number_of_elements*w_number_of_elements #%% Update with Number of Tasks




    #%% Export and Return Statements:

    return q_state


    #%% Documentation:

    # https://numpy.org/doc/stable/reference/generated/numpy.array.html
    # https://numpy.org/doc/stable/reference/generated/numpy.arange.html