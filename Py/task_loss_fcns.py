
def loss_lin_drop(t, w, t_start, t_drop, l_drop):

    # TODO: write fcn docstring

    # TODO: return a function???

    if t < t_start:
        loss = float("inf")
    elif (t >= t_start) and (t < t_drop):
        loss = w*(t-t_start)
    else:
        loss = l_drop

    if l_drop < w*(t_drop-t_start):
        raise ValueError
        # print('Error: Function is not monotonically non-decreasing')

    return loss
