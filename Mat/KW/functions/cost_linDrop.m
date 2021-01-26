function [c,w,t_start,t_drop,c_drop] = cost_linDrop(t,w,t_start,t_drop,c_drop)

if t < t_start
    c = Inf;
elseif (t >= t_start) && (t < t_drop)
    c = w*(t-t_start);
else
    c = c_drop;
end

if c_drop < w*(t_drop-t_start)
    disp('Error: Function is not monotonically non-descreasing');
    return
end