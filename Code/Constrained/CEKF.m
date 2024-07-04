function [Pxx_poster, x_poster, x_priori, phi] = CEKF(F, H, x_prior, u, Pxx_prior,...
    z, Q, R, Constraints, Jacob_x, Jacob_y, flag_stat, varargin)
% Calculation of the constrained extended Kalman filter (CEKF) based on the 
% methodology proposed by Gesthuisen et al. (2001)
%
% Inputs:
% F - State transition function
% H - Measurement function
% x_prior - State estimation based on information a posteriori at time k-1
% u - Set of time-invariant variables between sampling times k-1 and k
% Pxx_prior - State error covariance matrix at time k-1
% z - Measurements from the system at time k
% Q - Covariance matrix of the process noises
% R - Covariance matrix of the observation noises
% Constraints - Structure comprising the set of model constraints
% Jacob_x - Numerical calculation of the state transition matrix
% Jacob_y - Measurement matrix
% 
% Outputs:
% Pxx_poster - State error covariance matrix at time k
% x_poster - State estimation at time k
% x_priori - State estimation based on information a priori at time k
% phi - State transition matrix at time k
%
% References:
% Gesthuisen, R., Klatt, K. U. & Engell, S. Optimization-based state estimation
% — a comparative study for the batch polycondensation of polyethyleneterephthalate.
% In 2001 European Control Conference (ECC), 1062–1067, doi: 10.23919/ECC.2001.7076055 (2001).
%
% A. H. Jazwinski, Stochastic Processes and Filtering Theory, Mathematics in
% Science and Engineering, vol. 64, Academic Press, New York and London, 1970
%
% Simon, D. (2006). Optimal state estimation: Kalman, H infinity, and nonlinear
% approaches. John Wiley & Sons.
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The algorithm is equivalent to the extended Kalman filter,
% transcribed by Jazwinski (1970), if there are no active constraints in
% the projection function
%
% Note2: Gesthuisen et al. (2002) defined the CEKF for a time-invariant system;
% thus, following the formulation of an asymptotically stable or stationary
% system in the Kalman filter. The parameter flag_stat eases the switch between 
% time-invariant (flag_stat = 1) and time-varying (flag_stat = 0) versions.
%
% Note3: Simon (2006) detailed the assumptions and formulations of the steady
% state and time-varying Kalman filter in Sections 7.3 and 7.1, respectively.
%
% Note4: The variables x_priori and phi are returned to speed up computational
% time of a forward-backward filtering.

% Definition of the number of states and measurements
nx = numel(x_prior);
ny = numel(z);

% Propagation of states based on the dynamic model
x_priori = F(x_prior(1:nx),u);

% Calculation of the state transition matrix
phi = Jacob_x(x_prior(1:nx), u);

% Calculation of the a priori state error covariance matrix
if flag_stat == 1
    Pxx_priori = Pxx_prior; % Main assumption of the time-invariant formulation
else
    Pxx_priori = (phi*Pxx_prior*phi'+Q);
end

% Calculation of the Kalman gain
inv_K = mldivide(Jacob_y*Pxx_priori*Jacob_y'+R,eye(ny));
inv_K = (inv_K+inv_K')/2;
K = Pxx_priori*Jacob_y'*inv_K;

% Update of state error covariance matrix based on information a posteriori
if flag_stat == 1
    Pxx_poster = phi*Pxx_prior*phi'-phi*K*Jacob_y*Pxx_prior*phi'+Q; 
else
    Pxx_poster = (eye(nx)-K*Jacob_y)*Pxx_priori*(eye(nx)-K*Jacob_y)'+K*R*K'; 
end
Pxx_poster = (Pxx_poster+Pxx_poster')/2;

% Calculation of optimal unconstrained state estimations based on information a posteriori
x_poster = x_priori + K*(z-H(x_priori));

% Note: Proposal of Gesthuisen et al. (2001) refer only to an optimization 
% problem for flag_stat = 1. Hence, current implementation only follows it 
% under constraint violations.

% Definition of tolerances related to the optimization problem
TolX = 1e-6;
TolFun = 1e-6;
TolCon = 1e-8;

% Checks if x_poster from the unconstrained problem is within the feasible region
D = [];
d = [];
flag_nln = 0;
h_poster = Constraints.h(x_poster,u);
if ~isempty(h_poster)
    flag_nln = 1;
    D = [D; h_poster; -h_poster];
    d = [d; 0*h_poster; 0*h_poster];
    g_poster = Constraints.g(x_poster,u);
    if ~isempty(g_poster)
        D = [D; g_poster];
        d = [d; 0*g_poster];
        fun_g = @(w) Constraints.g(x_priori+w,u);
    else
        fun_g = @(w) [];
    end
    fun_h = @(w) Constraints.h(x_priori+w,u);
else
    g_poster = Constraints.g(x_poster,u);
    if ~isempty(g_poster)
        flag_nln = 1;
        D = [D; g_poster];
        d = [d; 0*g_poster];
        fun_g = @(w) Constraints.g(x_priori+w,u);
        fun_h = @(w) [];
    end
end
Dlin = [Constraints.Aeq(x_poster,u); -Constraints.Aeq(x_poster,u);...
    Constraints.A(x_poster,u); eye(nx);-eye(nx)];
dlin = [-Constraints.Beq(x_poster,u); -Constraints.Beq(x_poster,u);...
    Constraints.B(x_poster,u); Constraints.ubx; -Constraints.lbx];
D = [D; Dlin*x_poster];
d = [d; dlin];

% Solves a quadratic or nonlinear optimization problem if there were
% constraint violations
if any(D-d>TolCon)
    % Definition of coefficients from the objective function
    inv_Pxx = mldivide(Pxx_priori,eye(nx));
    inv_Pxx = (inv_Pxx+inv_Pxx')/2;
    inv_R = mldivide(R,eye(ny));
    inv_R = (inv_R+inv_R')/2;
    coef_quad = inv_Pxx + Jacob_y'*inv_R*Jacob_y;
    coef_quad = (coef_quad + coef_quad')/2;
    coef_lin = -Jacob_y'*inv_R*(z-H(x_priori));
    % OBS: The optimization problem is defined with respect to process noises.

    % Definition of linear constraints
    Aw = Dlin;
    Bw = dlin - Aw*x_priori;

    % Solves the optimization problem
    if flag_nln
        FO_nln = @(w) 0.5*w'*coef_quad*w+coef_lin'*w;
        opts = optimoptions('fmincon','Algorithm','sqp','TolX',TolX,'TolFun',TolFun,...
            'TolCon',TolCon,'MaxFunEvals',1e5,'MaxIter',1e5, 'Display', 'none');
        aux = fmincon(@(var) FO_nln(var), zeros(nx,1), Aw, Bw, [], [], [], [],...
            @(var) Restr_nln(var, fun_g, fun_h), opts);
    else
        opts = optimset('Algorithm','interior-point-convex','Display','none','TolX',TolX,...
            'TolFun', TolFun,'TolCon',TolCon,'MaxIter',1e5);
        aux = quadprog(coef_quad, coef_lin, Aw, Bw, [], [], [], [], x_priori,opts);
    end   

    % Update of the constrained state estimations based on the solution of 
    % the optimization problem
    x_poster = x_priori + aux;
end

end
function [c, ceq] = Restr_nln(w, fun_g, fun_h)
% Definition of nonlinear constraints according to the input format from MATLAB
%
% Inputs:
% w - Process noises
% fun_g - Nonlinear inequality constraints according to g(w) <= 0
% fun_h - Nonlinear equality constraints according to h(w) = 0
%
% Outputs:
% c - Nonlinear inequality constraints c <= 0
% ceq - Nonlinear inequality constraints ceq <= 0
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024

% Definition of the inputs and outputs sizes
c = fun_g(w);
ceq = fun_h(w);

end