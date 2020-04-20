function [c,w,t_start] = costPWlinear(t,w,t_start,t_drop,c_drop)

% Piece-wise linear cost function. 



% Inputs
% t - is the current time
% w - Nx1 vector of weights (assumed > 0)
% t_start - Nx1 vector of task starting times
% t_drop - Nx1 vector of dropping times
% c_drop - Nx1 vector of dropping penalities

% Outpus 
% c - Nx1 costs of performing tasks at various start times
N = length(w);
% c = w.*(t-t_start) + (t> (t_drop+ t_start) ).*c_drop;
c = w.*(t-t_start) + (t> (t_drop) ).*c_drop;  % Error in equation above KW 09APR2020


% if t < t_start
if N  > 1
    c(t<t_start) = Inf; % If current time is before task start time there is an infinite cost
else
   if t < t_start
       c = Inf; % KW - 3/17/20. This may need to be changed to -INF, may depend on application
%        keyboard
   end
end





