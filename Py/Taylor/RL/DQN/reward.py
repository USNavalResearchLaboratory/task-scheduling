def reward_function(cost,zero_cost_reward):

    #%% Reward:

    if cost == 0:
        reward = zero_cost_reward
    else:
        reward= 1/cost

    # print("\nReward = \n\n{}".format(reward))


    #%% Export and Return Statements:

    return reward