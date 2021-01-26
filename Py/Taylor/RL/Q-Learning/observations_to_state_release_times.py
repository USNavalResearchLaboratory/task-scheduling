def observations_to_state_release_times_function(number_of_tasks,r_continuous,r_N_steps):

    # print("\n\nExecuting 'observations_to_state_release_times.py' ...")


    #%% Import Statements:

    import numpy


    #%% Mapping Observations to State:

    r_N_low = 0
    r_N_high = 1
    r_N_dx = (r_N_high-r_N_low)/r_N_steps

    r_discrete = numpy.zeros([len(r_continuous)])

    for i in range(len(r_discrete)):
        r_discrete[i] = int((r_continuous[i]-r_N_low)/r_N_dx)

    # r_number_of_elements = r_N_steps**number_of_tasks
    # r_states = numpy.array(numpy.arange(0,r_number_of_elements)).reshape(r_N_steps,r_N_steps,r_N_steps) #%% Update with Number of Tasks
    # r_state = r_states[int(r_discrete[0]),int(r_discrete[1]),int(r_discrete[2])] #%% Update with Number of Tasks

    r_state = int(r_discrete[0])+int(r_discrete[1])*r_N_steps+int(r_discrete[2])*r_N_steps**2 #%% Update with Number of Tasks
    

    #%% Export and Return Statements:

    return r_state


    #%% Documentation:

    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # https://numpy.org/doc/stable/reference/generated/numpy.array.html
    # https://numpy.org/doc/stable/reference/generated/numpy.arange.html