def training_function(number_of_tasks,number_of_states,F,reward_matrix,episodes,alpha,gamma,epsilon):

    print("\n\nExecuting 'training.py' ...")


    ##### Import Statements #####

    from observations_to_state import observations_to_state_function

    import math
    import numpy
    import matplotlib.pyplot as plt

    ##### Training (Q-Learning) #####

    r_number_of_states = number_of_states[0]
    w_number_of_states = number_of_states[1]
    l_number_of_states = number_of_states[2]

    Q = numpy.zeros([r_number_of_states**3,w_number_of_states**3,l_number_of_states**3,math.factorial(number_of_tasks)])

    print("\n\tShape of Q-Matrix = {}".format(numpy.shape(Q)))

    cum_reward = numpy.zeros(episodes+1)
    for i in range(1,episodes+1):

        for j in range(len(F)-1):

            r_continuous = F[j,0:1*number_of_tasks]
            w_continuous = F[j,1*number_of_tasks:2*number_of_tasks]
            l_continuous = F[j,2*number_of_tasks:3*number_of_tasks]

            r_state,w_state,l_state = observations_to_state_function(r_continuous,w_continuous,l_continuous,r_number_of_states,w_number_of_states,l_number_of_states)

            if numpy.random.uniform(0,1) < epsilon:
                action = numpy.random.randint(math.factorial(number_of_tasks))
            else:
                action = numpy.argmax(Q[r_state][w_state][l_state][:])

            old_value = Q[r_state][w_state][l_state][action]

            reward = reward_matrix[j][action]

            r_continuous_next = F[j+1,0:1*number_of_tasks]
            w_continuous_next = F[j+1,1*number_of_tasks:2*number_of_tasks]
            l_continuous_next = F[j+1,2*number_of_tasks:3*number_of_tasks]

            r_state_next,w_state_next,l_state_next = observations_to_state_function(r_continuous_next,w_continuous_next,l_continuous_next,r_number_of_states,w_number_of_states,l_number_of_states)

            maximum_next = numpy.max(Q[r_state_next][w_state_next][l_state_next][:])

            new_value = (1-alpha)*old_value+alpha*(reward+gamma*maximum_next)

            Q[r_state][w_state][l_state][action] = new_value

            try:
                cum_reward[i] = cum_reward[i] + reward
            except:
                print("An exception occurred")

            # if i == len(F)-1:
            #     print("\nAction = {}".format(action))
            #     print("Reward = {}".format(reward))
            #     print("Next Maximum = {}".format(maximum_next))
            #     print("Old Value = {}".format(old_value))
            #     print("New Value = {}".format(new_value))
            #     print("Q = {}".format(Q))

        if i % 10 == 0:
            print(f"\n\tEpisode: {i}")



    print("\n\tTRAINING FINISHED")
    plt.figure(1)
    plt.plot(cum_reward)
    plt.show()

    ##### Export and Return Statements #####

    return Q


    ##### Documentation #####

    # https://docs.python.org/3/library/math.html

    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # https://numpy.org/doc/1.18/reference/generated/numpy.shape.html
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randint.html
    # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    # https://numpy.org/doc/stable/reference/generated/numpy.maximum.html