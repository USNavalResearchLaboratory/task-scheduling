def training_function(number_of_tasks,features_indices,number_of_steps,F_training,F_testing,permutations,zero_cost_reward,episodes,alpha,gamma,epsilon):

    print("\n\nExecuting 'training.py' ...")


    #%% Import Statements:

    from observations_to_state import observations_to_state_function
    from observations_to_state_equation import observations_to_state_equation_function
    from cost import cost_function
    from reward import reward_function
    from testing import testing_function

    import math
    import numpy


    #%% Definitions of Variables Based on User-Specified Inputs:

    r_N_index = features_indices[0]
    w_N_index = features_indices[1]
    l_N_index = features_indices[2]

    r_N_steps = number_of_steps[0]
    w_N_steps = number_of_steps[1]
    l_N_steps = number_of_steps[2]

    r_N_training = F_training[:,(r_N_index-1)*number_of_tasks:(r_N_index*number_of_tasks)]
    w_N_training = F_training[:,(w_N_index-1)*number_of_tasks:(w_N_index*number_of_tasks)]
    l_N_training = F_training[:,(l_N_index-1)*number_of_tasks:(l_N_index*number_of_tasks)]

    r_N_testing = F_testing[:,(r_N_index-1)*number_of_tasks:(r_N_index*number_of_tasks)]
    w_N_testing = F_testing[:,(w_N_index-1)*number_of_tasks:(w_N_index*number_of_tasks)]
    l_N_testing = F_testing[:,(l_N_index-1)*number_of_tasks:(l_N_index*number_of_tasks)]


    #%% Training (Q-Learning):

    Q = numpy.zeros([r_N_steps**number_of_tasks,w_N_steps**number_of_tasks,l_N_steps**number_of_tasks,math.factorial(number_of_tasks)])
    # print("\n\tShape of Q-Matrix = {}".format(numpy.shape(Q)))

    total_loss_training_over_training = numpy.zeros([episodes])
    # print("\n\tShape of Total Loss Matrix (Training) = {}".format(numpy.shape(total_loss_training_over_training)))

    total_reward_training_over_training = numpy.zeros([episodes])
    # print("\n\tShape of Total Reward Matrix (Training) = {}".format(numpy.shape(total_reward_training_over_training)))

    total_loss_testing_over_training = numpy.zeros([episodes])
    # print("\n\tShape of Total Loss Matrix (Testing) = {}".format(numpy.shape(total_loss_testing_over_training)))

    total_reward_testing_over_training = numpy.zeros([episodes])
    # print("\n\tShape of Total Reward Matrix (Testing) = {}".format(numpy.shape(total_reward_testing_over_training)))

    for i in range(1,episodes+1):

        loss_training = numpy.zeros([len(F_training)-1])
        reward_training = numpy.zeros([len(F_training)-1])

        for j in range(len(F_training)-1):

            r_continuous = r_N_training[j,:]
            w_continuous = w_N_training[j,:]
            l_continuous = l_N_training[j,:]

            # r_state,w_state,l_state = observations_to_state_function(number_of_tasks,r_continuous,w_continuous,l_continuous,r_N_steps,w_N_steps,l_N_steps)
            r_state,w_state,l_state = observations_to_state_equation_function(number_of_tasks,r_continuous,w_continuous,l_continuous,r_N_steps,w_N_steps,l_N_steps)

            epsilon_minimum = 0.001

            epsilon_decreasing = max(epsilon_minimum,epsilon*(0.85**(i//10)))

            if numpy.random.uniform(0,1) < epsilon_decreasing:
                action = numpy.random.randint(math.factorial(number_of_tasks))
            else:
                action = numpy.argmax(Q[r_state][w_state][l_state][:])

            old_value = Q[r_state][w_state][l_state][action]

            sequence = permutations[action,:]

            r_continuous_reshape = numpy.reshape(r_continuous,(1,number_of_tasks))
            w_continuous_reshape = numpy.reshape(w_continuous,(1,number_of_tasks))
            l_continuous_reshape = numpy.reshape(l_continuous,(1,number_of_tasks))
            sequence_reshape = numpy.reshape(sequence,(1,number_of_tasks))

            cost = cost_function(r_continuous_reshape,w_continuous_reshape,l_continuous_reshape,sequence_reshape)

            loss_training[j] = cost

            reward = reward_function(cost,zero_cost_reward)

            reward_training[j] = reward

            r_continuous_next = r_N_training[j+1,:]
            w_continuous_next = w_N_training[j+1,:]
            l_continuous_next = l_N_training[j+1,:]

            # r_state_next,w_state_next,l_state_next = observations_to_state_function(number_of_tasks,r_continuous_next,w_continuous_next,l_continuous_next,r_N_steps,w_N_steps,l_N_steps)
            r_state_next,w_state_next,l_state_next = observations_to_state_equation_function(number_of_tasks,r_continuous_next,w_continuous_next,l_continuous_next,r_N_steps,w_N_steps,l_N_steps)

            maximum_next = numpy.max(Q[r_state_next][w_state_next][l_state_next][:])

            alpha_minimum = 0.001

            alpha_decreasing = max(alpha_minimum,alpha*(0.85**(i//10)))

            new_value = (1-alpha_decreasing)*old_value+alpha_decreasing*(reward+gamma*maximum_next)

            Q[r_state][w_state][l_state][action] = new_value

            # print("\nAction = {}".format(action))
            # print("Reward = {}".format(reward))
            # print("Next Maximum = {}".format(maximum_next))
            # print("Old Value = {}".format(old_value))
            # print("New Value = {}".format(new_value))

        total_loss_training_over_training[i-1] = numpy.sum(loss_training)
        total_reward_training_over_training[i-1] = numpy.sum(reward_training)

        results = testing_function(number_of_tasks,features_indices,number_of_steps,F_testing,Q)

        sequences = numpy.zeros([len(results),number_of_tasks])

        for k in range(len(results)):

            action = results[k]

            sequence = permutations[action.astype('int'),:]

            sequences[k,:] = sequence

        loss_testing = cost_function(r_N_testing,w_N_testing,l_N_testing,sequences)

        reward_testing = numpy.zeros([len(results)])

        for k in range(len(loss_testing)):
            reward_testing[k] = reward_function(loss_testing[k],zero_cost_reward)

        total_loss_testing = numpy.sum(loss_testing)
        total_reward_testing = numpy.sum(reward_testing)

        total_loss_testing_over_training[i-1] = total_loss_testing
        total_reward_testing_over_training[i-1] = total_reward_testing

        if i % 10 == 0:
            print(f"\n\tEpisode: {i}")

    print("\n\tTRAINING FINISHED")


    #%% Export and Return Statements:

    return Q, total_loss_training_over_training, total_reward_training_over_training, total_loss_testing_over_training, total_reward_testing_over_training


    #%% Documentation:

    # https://docs.python.org/3/library/math.html

    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # https://numpy.org/doc/stable/reference/generated/numpy.shape.html
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
    # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    # https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
    # https://numpy.org/doc/stable/reference/generated/numpy.sum.html