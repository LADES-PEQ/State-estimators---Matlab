function output = solvingODE(fun,dt,X0,U,OPTIONS)
[~,x] = ode15s(@(t,x) fun(t,x,U),[0 dt],X0,OPTIONS);
output = x(end,:)';
end