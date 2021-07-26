def analysis_EST_function(results_EST,results):

    #%% Import Statements:

    import numpy


    #%% Analysis of Results:

    print("\nResults (EST) = \n\n{}".format(results_EST))
    print("\nShape of Results (EST) = {}".format(numpy.shape(results_EST)))

    print("\nResults (RL) = \n\n{}".format(results))
    print("\nShape of Results (RL) = {}".format(numpy.shape(results)))

    check = numpy.zeros(len(results_EST))

    counter = 0

    for i in range(len(results_EST)):

        ideal = results_EST[i]
        prediction = results[i]

        if ideal == prediction:
            check[i] = 1
            counter += 1
        else:
            check[i] = 0

    # print("\nCheck = \n\n{}".format(check))
    # print("\n\tShape of Check = {}".format(numpy.shape(check)))

    # print("\n\tCount = {}".format(counter))

    accuracy_check = counter/len(check)

    print("\nAccuracy (Comparison with EST) = {:.0%}".format(accuracy_check))


    #%% Export and Return Statements:

    return accuracy_check


    #%% Documentation:

    # https://numpy.org/doc/stable/reference/generated/numpy.shape.html
    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html