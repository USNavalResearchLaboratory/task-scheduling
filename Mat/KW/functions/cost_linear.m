function [c,w,t_start] = cost_linear(t,w,t_start)

% Inputs
% t - is the current time
% w - Nx1 vector of weights (assumed > 0)
% t_start - Nx1 vector of task starting times

% Outpus 
% c - Nx1 costs of performing tasks at various start times
N = length(w);
c = w.*(t-t_start);

% if t < t_start
if N  > 1
    c(t<t_start) = Inf; % If current time is before task start time there is an infinite cost
else
   if t < t_start
       c = Inf;
%        keyboard
   end
end





