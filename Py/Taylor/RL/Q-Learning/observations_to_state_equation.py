def observations_to_state_equation_function(number_of_tasks,r_continuous,w_continuous,l_continuous,r_N_steps,w_N_steps,l_N_steps):

    # print("\n\nExecuting 'observations_to_state.py' ...")


    #%% Import Statements:

    import numpy


    #%% Mapping Observations to State:

    r_N_low = 0
    r_N_high = 1
    r_N_dx = (r_N_high-r_N_low)/r_N_steps

    w_N_low = 0
    w_N_high = 10
    w_N_dx = (w_N_high-w_N_low)/w_N_steps

    l_N_low = 0
    l_N_high = 1
    l_N_dx = (l_N_high-l_N_low)/l_N_steps

    r_discrete = numpy.zeros([len(r_continuous)])
    w_discrete = numpy.zeros([len(w_continuous)])
    l_discrete = numpy.zeros([len(l_continuous)])

    for i in range(len(r_discrete)):
        r_discrete[i] = int((r_continuous[i]-r_N_low)/r_N_dx)
        w_discrete[i] = int((w_continuous[i]-w_N_low)/w_N_dx)
        l_discrete[i] = int((l_continuous[i]-l_N_low)/l_N_dx)

    r_state = int(r_discrete[0])+int(r_discrete[1])*r_N_steps+int(r_discrete[2])*r_N_steps**2 #%% Update with Number of Tasks
    w_state = int(w_discrete[0])+int(w_discrete[1])*w_N_steps+int(w_discrete[2])*w_N_steps**2 #%% Update with Number of Tasks
    l_state = int(l_discrete[0])+int(l_discrete[1])*l_N_steps+int(l_discrete[2])*l_N_steps**2 #%% Update with Number of Tasks


    #%% Export and Return Statements:

    return r_state,w_state,l_state


    #%% Documentation:

    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html