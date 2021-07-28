def n_dimensional_regression_coefficients_results_function(number_of_tasks,number_of_steps):

    #%% Coefficients for N-Dimensional Regression:

    if number_of_tasks <= 2: #%% Assumption: Minimum = Three Tasks
        print("\n\tWARNING: Coefficients Not Defined\n")

    elif number_of_tasks == 3:
        if number_of_steps <= 2: #%% Assumption: Minimum = Three Steps
            print("\n\tWARNING: Coefficients Not Defined\n")
        elif number_of_steps == 3:
            # B = [[3.0266234e-11],[7.2900000e+02],[2.7000000e+01],[1.0000000e+00]] #%% 1000000 Sets
            B = [[3.46185303e-11],[1.00000000e+00],[2.70000000e+01],[7.29000000e+02]] #%% 1000000 Sets
        elif number_of_steps == 4:
            B = [[-1.09935172e-10],[4.09600000e+03],[6.40000000e+01],[1.00000000e+00]] #%% 1000000 Sets
        elif number_of_steps == 5:
            B = [[-5.90745231e-10],[1.56250000e+04],[1.25000000e+02],[1.00000000e+00]] #%% 1000000 Sets
        elif number_of_steps >= 6: #%% Assumption: Maximum = 6 Steps (Otherwise...FINISH)
            print("\n\tWARNING: Coefficients Not Defined\n")

    elif number_of_tasks == 4:
        if number_of_steps <= 2: #%% Assumption: Minimum = Three Steps
            print("\n\tWARNING: Coefficients Not Defined\n")
        elif number_of_steps == 3:
            B = [[-3.71216946e-10],[6.56100000e+03],[8.10000000e+01],[1.00000000e+00]] #%% 1000000 Sets
        elif number_of_steps == 4:
            B = [[7.28749683e-09],[6.55360000e+04],[2.56000000e+02],[1.00000000e+00]] #%% 1000000 Sets
        # elif number_of_steps == 5: #%% FINISH
            # B =  #%% FINISH
        elif number_of_steps >= 6: #%% Assumption: Maximum = 6 Steps (Otherwise...FINISH)
            print("\n\tWARNING: Coefficients Not Defined\n")
    
    elif number_of_tasks >= 5:
        print("\n\tWARNING: Coefficients Not Defined\n") #%% Assumption: Maximum = Four Tasks (Otherwise...FINISH)
            

    #%% Export and Return Statements:

    return B