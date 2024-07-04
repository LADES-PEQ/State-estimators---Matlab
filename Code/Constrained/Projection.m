function [x_proj] = Projection(x, u, Constraints, varargin)
% Search violated constraints for a given state x and project the violations
% found to the boundary of the feasible region based on the clipping methodology
% proposed by Simon and Chia (2002)
%
% Inputs:
% x - State
% Constraints - Structure comprising the set of model constraints
% Pxx_poster - State error covariance matrix
%
% Outputs:
% x_proj - Projected state at the boundary of the feasible region
%
% References:
% D. Simon, T.L. Chia, Kalman filtering with state equality constraints,
% IEEE Transactions on Aerospace and Electronic Systems 38 (2002) 128–136.
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note: This function projects x based on the mean square method if Pxx_poster
% is not provided. Otherwise, the projection follows the maximum probability method.

% Definition of the number of states
nx = numel(x);
Tol = 1e-6;
g_lin = [];

% Definition of the weighting matrix W
Pxx_poster = varargin;
if isempty(Pxx_poster)
    % Least square minimization
    inv_W = eye(nx);
else
    % Maximum probability
    inv_W = Pxx_poster{1};
end
% inv_W(abs(inv_W)<eps)=eps;

x_proj = x;
x_in = x;

% Memory allocation for the coefficients of the projected equality D*x=d
D = Constraints.Aeq(x,u);
d = Constraints.Beq(x,u);

% % Checks for nonlinear equality constraints
h_poster = Constraints.h(x,u);
if ~isempty(h_poster)
    if any(abs(h_poster) > Tol)
        h_lin = JacNum(x, u, Constraints.h);
        D = [D; h_lin];
        d = [d; -h_poster+h_lin*x];
        x_proj = x_in-inv_W*D'/(D*inv_W*D')*(D*x_in-d); % Eq. 37
    end
end

it = 0;
while true

% Checks for violations of linear inequality constraints
A = Constraints.A(x,u);
B = Constraints.B(x,u);
if ~isempty(A) && ~isempty(B)
    pos_viol = A*x - B > Tol;
    if ~isempty(pos_viol)
        D = [D; A(pos_viol,:)];
        d = [d; B(pos_viol)];
    end
end

% Checks for violations of nonlinear inequality constraints
g = Constraints.g(x,u);
if ~isempty(g)
    pos_viol = g>Tol;
    if any(pos_viol) 
        if isempty(g_lin)
            g_lin = JacNum(x_in, u, Constraints.g);
        end
        g_x = Constraints.g(x_in, u);
        D = [D; g_lin(pos_viol,1:nx)];
        d = [d; -g_x(pos_viol)+g_lin(pos_viol,1:nx)*x_in];
    end
end

% % Checks for upper boundary violations
if ~isempty(Constraints.ubx)
    pos_viol = x - Constraints.ubx > Tol;
    if any(pos_viol)
        A_ubx = eye(nx);
        A_ubx = A_ubx(pos_viol,:);
        D = [D; A_ubx];
        d = [d; Constraints.ubx(pos_viol)];
    end
end

% Checks for lower boundary violations
if ~isempty(Constraints.lbx)
    pos_viol = - x + Constraints.lbx > Tol;
    if any(pos_viol)
        A_ubx = eye(nx);
        A_ubx = A_ubx(pos_viol,:);
        D = [D; -A_ubx];
        d = [d; -Constraints.lbx(pos_viol)];
    end
end

d(d==0) = 0;
% Checks if a projection is required
if ~isempty(D)
    D2 = D;
    d2 = d;
    step = 0.001;
    dif2 = D*x-d;
    it2 = 0;
    while true
        if ~any(D2*x_proj-d2>Tol)
            break
        end
        if it2>10000
            break;
        end
        inv_proj = pinv(D*inv_W*D');
        inv_proj = (inv_proj+inv_proj')/2;
        x_proj = x_in-inv_W*D'*inv_proj*(D*x_in-d); % Eq. 37
        
        dif = D2*x_proj-d2;
        pos_viol = dif>Tol;
        if norm(d)>1/eps || norm(D)>1/eps
            D = D*eps;
            d = d*eps;
        end
        D(pos_viol,:) = (1+step+dif(pos_viol,:)).*D(pos_viol,:);
        d(pos_viol,:) = (1+step+dif(pos_viol,:)).*d(pos_viol,:)-Tol;
        it2 = it2+1;
    end
    D = D2;
    d = d2;
else
    % Returns the state unchanged
    x_proj = x;
    break;
end

if any(abs(x - x_proj)>Tol)
    x = x_proj;
else
    x = x_proj; 
    break;
end
it = it+1;
if it == 1000
    break;
end

end

end
function dFundx = JacNum(x, u, Fun)
% Numerical calculation of the Jacobian of a function 'Fun' with respect to
% a variable 'x' based on central finite differences
%
% Inputs:
% x - Variable
% Fun - Function handle
%
% Outputs:
% dFundx - Jacobian of Fun with respect to x
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024

% Definition of the inputs and outputs sizes
nx = numel(x);
ny = numel(Fun(x,u));
dx = zeros(nx,1);
dFundx = zeros(ny,nx);

% Jacobian calculation
for i = 1:nx
    dx = zeros(nx,1);
    dx(i) = max(1e-6*x(i),1e-7);
    aux = Fun(x+dx(1:nx,1),u);
    aux2 = Fun(x-dx(1:nx,1),u);
    dFundx(1:ny,i) = (aux-aux2)/(2*dx(i));
end

end