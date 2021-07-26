def observations_to_state_weights_function(number_of_tasks,w_continuous,w_N_steps):

    # print("\n\nExecuting 'observations_to_state_weights.py' ...")


    #%% Import Statements:

    import numpy


    #%% Mapping Observations to State:

    w_N_low = 0
    w_N_high = 10
    w_N_dx = (w_N_high-w_N_low)/w_N_steps

    w_discrete = numpy.zeros([len(w_continuous)])

    for i in range(len(w_discrete)):
        w_discrete[i] = int((w_continuous[i]-w_N_low)/w_N_dx)

    # w_number_of_elements = w_N_steps**number_of_tasks
    # w_states = numpy.array(numpy.arange(0,w_number_of_elements)).reshape(w_N_steps,w_N_steps,w_N_steps) #%% Update with Number of Tasks
    # w_state = w_states[int(w_discrete[0]),int(w_discrete[1]),int(w_discrete[2])] #%% Update with Number of Tasks

    w_state = int(w_discrete[0])+int(w_discrete[1])*w_N_steps+int(w_discrete[2])*w_N_steps**2 #%% Update with Number of Tasks


    #%% Export and Return Statements:

    return w_state


    #%% Documentation:

    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # https://numpy.org/doc/stable/reference/generated/numpy.array.html
    # https://numpy.org/doc/stable/reference/generated/numpy.arange.html