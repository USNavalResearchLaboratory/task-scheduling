def input_output_pairs_function(number_of_sets,number_of_tasks,number_of_features,maximum_weight_of_tasks):

    ##### Import Statements #####

    import numpy
    import pandas

    from task_generation_with_EST_algorithm import task_generation_with_EST_algorithm_function


    ##### Data (Fix "N" and Generate "MC" Sets of Tasks) #####

    inputs_before = task_generation_with_EST_algorithm_function(number_of_sets,number_of_tasks,number_of_features,maximum_weight_of_tasks)
    print("\n\nExecuting 'input_output_pairs.py' ...")

    # print("\nInputs (Before) = \n\n{}".format(inputs_before))
    print("\n\tShape of Inputs (Before) = {}".format(numpy.shape(inputs_before)))


    ##### SL Input-Output Pairs ##### !!!!! UPDATE AS TASKS AND FEATURES UPDATE !!!!!

    X = numpy.zeros([(number_of_features)+number_of_tasks,(number_of_sets*number_of_tasks*number_of_tasks)])
    Y = numpy.zeros([1,number_of_sets*number_of_tasks])

    for i in range(len(inputs_before)):

        features = numpy.zeros([number_of_features,number_of_tasks])

        release_times = inputs_before[i,0:1*number_of_tasks]
        features[0,:] = release_times

        weights = inputs_before[i,number_of_tasks:2*number_of_tasks]
        features[1,:] = weights

        lengths = inputs_before[i,2*number_of_tasks:3*number_of_tasks]
        features[2,:] = lengths

        # print("\nF = \n\n{}".format(features))

        EST = inputs_before[i,3*number_of_tasks:4*number_of_tasks]

        decisions_1 = numpy.zeros([number_of_tasks,number_of_tasks])

        # print("\nD(1) = \n\n{}".format(decisions_1))

        decisions_2 = numpy.zeros([number_of_tasks,number_of_tasks])
        decisions_2[0,EST[0].astype('int')] = 1

        # print("\nD(2) = \n\n{}".format(decisions_2))

        decisions_3 = numpy.zeros([number_of_tasks,number_of_tasks])
        decisions_3[0,EST[0].astype('int')] = 1
        decisions_3[1,EST[1].astype('int')] = 1

        # print("\nD(3) = \n\n{}".format(decisions_3))

        X_1 = numpy.concatenate((features,decisions_1),axis=0)
        X_2 = numpy.concatenate((features,decisions_2),axis=0)
        X_3 = numpy.concatenate((features,decisions_3),axis=0)

        X_incomplete = numpy.concatenate((X_1,X_2),axis=1)
        X_complete = numpy.concatenate((X_incomplete,X_3),axis=1)

        # print("\nX = \n\n{}".format(X_complete))

        X[:,(number_of_tasks*number_of_tasks*i):(number_of_tasks*number_of_tasks*(i+1))] = X_complete
        Y[:,(number_of_tasks*i):(number_of_tasks*(i+1))] = EST

    # print("\nX = \n\n{}".format(X))
    print("\n\tShape of X = {}".format(numpy.shape(X)))
    # print("\nY = \n\n{}".format(Y))
    print("\n\tShape of Y = {}".format(numpy.shape(Y)))


    ##### Export and Return Statements #####

    inputs_after = X
    outputs_after = Y

    # numpy.savetxt('SL_Inputs.txt',inputs_after,fmt='%.3e')
    # numpy.savetxt('SL_Outputs.txt',outputs_after,fmt='%.3e')

    return inputs_after, outputs_after


    ##### Documentation #####

    # https://numpy.org/doc/stable/reference/generated/numpy.shape.html
    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
    # https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html