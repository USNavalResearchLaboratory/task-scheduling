def observations_to_state_lengths_function(number_of_tasks,l_continuous,l_N_steps):

    # print("\n\nExecuting 'observations_to_state_lengths.py' ...")


    #%% Import Statements:

    import numpy


    #%% Mapping Observations to State:

    l_N_low = 0
    l_N_high = 1
    l_N_dx = (l_N_high-l_N_low)/l_N_steps

    l_discrete = numpy.zeros([len(l_continuous)])

    for i in range(len(l_discrete)):
        l_discrete[i] = int((l_continuous[i]-l_N_low)/l_N_dx)

    # l_number_of_elements = l_N_steps**number_of_tasks
    # l_states = numpy.array(numpy.arange(0,l_number_of_elements)).reshape(l_N_steps,l_N_steps,l_N_steps) #%% Update with Number of Tasks
    # l_state = l_states[int(l_discrete[0]),int(l_discrete[1]),int(l_discrete[2])] #%% Update with Number of Tasks

    l_state = int(l_discrete[0])+int(l_discrete[1])*l_N_steps+int(l_discrete[2])*l_N_steps**2 #%% Update with Number of Tasks


    #%% Export and Return Statements:

    return l_state


    #%% Documentation:

    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # https://numpy.org/doc/stable/reference/generated/numpy.array.html
    # https://numpy.org/doc/stable/reference/generated/numpy.arange.html