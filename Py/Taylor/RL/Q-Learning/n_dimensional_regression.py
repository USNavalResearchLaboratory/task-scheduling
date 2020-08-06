def n_dimensional_regression_function(X,Y):

    # print("\n\nExecuting 'n_dimensional_regression.py' ...")


    #%% Import Statements:

    import numpy


    #%% N-Dimensional Regression:

    X_transpose = numpy.transpose(X)

    X_transpose_X = numpy.matmul(X_transpose,X)

    X_transpose_X_inverse = numpy.linalg.inv(X_transpose_X)

    x_transpose_X_inverse_X_transpose = numpy.matmul(X_transpose_X_inverse,X_transpose)

    x_transpose_X_inverse_X_transpose_Y = numpy.matmul(x_transpose_X_inverse_X_transpose,Y)

    B = x_transpose_X_inverse_X_transpose_Y


    #%% Export and Return Statements:

    return B


    #%% Documentation:

    # https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
    # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html