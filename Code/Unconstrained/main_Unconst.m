% Main script of the work submitted to PSEBR 2024 entitled "Assessment of
% forward and forward-backward Bayesian filters for a Van de Vusse reactor"
%
% Outputs:
% Plant - Structure comprising the dynamics of the system
% Plant.T - Sampling time
% Plant.X - System states with process noises
% Plant.Y - System measurements without observation noises
% Plant.Z - System measurements with observation noises
% Plant.Par - Model parameters
% Results - Structure comprising the simulation results
% Results.X_sim - Real-time state estimations within the analysis period
% Results.Y_sim - Real-time measurement predictions within the analysis period
% Results.Pxx_sim - Real-time state error covariance matrices within the analysis period
% Results.X_vector - Moving horizon estimations within the analysis period
% Results.Pxx_vector - Moving horizon state error covariance matrices within the analysis period
% Results.Esf_vector - Observer execution time within the analysis period
% Results.Esf_mean - Average observer execution time for each simulated scenario
% Results.flag_est - Observer algorithm implemented in each simulated scenario
% Results.N_smo - Smoothing horizon for each simulated scenario
% Results.N_points - Number of state points calculated by the observer for each simulated scenario
% Results.MSE_x - Mean squared error (MSE) of state estimations for each simulated scenario
% Results.MAPE_x - Mean absolute percentage error (MAPE) of state estimations for each simulated scenario
% Results.MSE_y - MSE of measurement predictions for each simulated scenario
% Results.MAPE_y - MAPE of measurement predictions for each simulated scenario 
%
% References:
% Engell, S., & Klatt, K. U. (1993, June). Nonlinear control of a non-minimum-phase
% CSTR. In 1993 American Control Conference (pp. 2941-2945). IEEE.
%
% van de Vusse, J. G. (1964) Plug-flow type reactor versus
% tank reactor. Chem. Engng Sci. 19(6), 994--998
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024

clear all;
close all;

% Definition of the data directory
dir_save = "..\..\Results\Unconstrained\";
if ~exist(dir_save,'dir')
    mkdir(dir_save);
end

% Definition of ODE solver configurations
optsODE = [];

% Sampling time
Ts = 1/60;

% Definition of model parameters
Par_fix = [1.287*1e12; 1.287*1e12; 9.043*1e9; 9758.3; 9758.3; 8560; 4.2; -11;...
    -41.85; 0.9342; 3.01; 2.0; 0.215; 10.01; 5; 403.15; 4032];

% Definition of standard deviations of model parameters according to Engel
% and Klatt (1993)
w_par = [0.04*1e12; 0.04*1e12; 0.27*1e9; 0; 0; 0; 2.36; 1.92;...
    1.41; 0.0004; 0.04; 0.05; 0; 0; 0; 0; 120];

% Definition of the initial condition based on the standard condition reported
% by Engel and Klatt (1993)
X0 = [1.23458465484989;0.899860981545301;407.288267319750;402.098267321517];
U0 = [188.441436291548;-4499.10719846814;5.09938869604233];
% OBS: The previous values were calculated to establish the initial condition
% at the steady-state based on solving var_ss =  fsolve(@(var)
% ODE_VanderVusse(0,var_ss(1:nx),[var_ss(nx+1:nx+nu);Par_fix],var_guess),
% where var_guess follow the values of Table 1 from Engel and Klatt (1993).
Y0 = X0(2:3);

% Definition of the set of time-invariant model parameters
Par = [U0;Par_fix];
Par0 = Par;

% Definition of the number of states, inputs, and model parameters
nx = numel(X0);
nu = numel(U0);
npar = numel(Par);

% Definition of the measurement matrix
Jacob_y0 = zeros(2,nx);
Jacob_y0(1,2) = 1;
Jacob_y0(2,3) = 1;
ny = size(Jacob_y0,1);

% Definition of the simulation analysis period
T = 0:Ts:2; % [h]
% Number of sampling times in each simulation
N_k = numel(T);

% Definition of parameter and measurements noises
w_U = [zeros(nu,1);w_par];
vk = [0.025; 1];

% Definition of the state transition and measurement functions
F_real = @(var,Par) [solvingODE(@(t,x,u) ODE_VanderVusse(t,x,u), Ts,...
    var, Par, optsODE)];
H = @(x0) x0(2:3);

% Calculation of the Jacobian with respect to model parameters
dF_dPar = zeros(nx,npar);
for i = 1:npar
    du = zeros(npar,1);
    dx = zeros(nx,1);
    du(i) = max(1e-6*Par(i),1e-7);
    aux = F_real(X0+dx(1:nx,1),Par(1:npar,1)+du(1:npar,1));
    aux2 = F_real(X0-dx(1:nx,1),Par(1:npar,1)-du(1:npar,1));
    dF_dPar(1:nx,i) = (aux-aux2)/(2*du(i));
end

% Definition of the observer for each simulated scenarios
n_scen = 8*6; % Number of different scenarios
% OBS: Eight conditions evaluated for each observer
n_sim_scen = 50; % Number of simulations for each scenario
n_sim = 1+n_sim_scen*n_scen; % Total number of simulations
% OBS: First simulation is discarded to compensate the just-in-time compiler

N_smo = zeros(1,n_sim); % Smoothing or moving horizon
flag_est = zeros(1,n_sim); % Definition of the observer algorithm
N_points = zeros(1,n_sim); % Number of particles or ensemble members,
% depending on the observer
flag_est(1:8*n_sim_scen+1) = 1; % EKF&S
flag_est(8*n_sim_scen+2:16*n_sim_scen+1) = 2; % UKF&S
flag_est(16*n_sim_scen+2:24*n_sim_scen+1) = 3; % CKF&S
flag_est(24*n_sim_scen+2:32*n_sim_scen+1) = 4; % PF&S
flag_est(32*n_sim_scen+2:40*n_sim_scen+1) = 5; % EnKF&S
for i = [1:3,6]
    N_smo(((i-1)*8+0)*n_sim_scen+2:((i-1)*8+1)*n_sim_scen+1) = 0;
    N_smo(((i-1)*8+1)*n_sim_scen+2:((i-1)*8+2)*n_sim_scen+1) = 1;
    N_smo(((i-1)*8+2)*n_sim_scen+2:((i-1)*8+3)*n_sim_scen+1) = 2;
    N_smo(((i-1)*8+3)*n_sim_scen+2:((i-1)*8+4)*n_sim_scen+1) = 4;
    N_smo(((i-1)*8+4)*n_sim_scen+2:((i-1)*8+5)*n_sim_scen+1) = 6;
    N_smo(((i-1)*8+5)*n_sim_scen+2:((i-1)*8+6)*n_sim_scen+1) = 8;
    N_smo(((i-1)*8+6)*n_sim_scen+2:((i-1)*8+7)*n_sim_scen+1) = 10;
    N_smo(((i-1)*8+7)*n_sim_scen+2:((i-1)*8+8)*n_sim_scen+1) = 15;
end
% OBS: Smoothing horizon is defined only for EKF&S, UKF&S, and CKF&S
for i = 1:2
    N_points(((i-1)*8+24)*n_sim_scen+2:((i-1)*8+25)*n_sim_scen+1) = 10;
    N_points(((i-1)*8+25)*n_sim_scen+2:((i-1)*8+26)*n_sim_scen+1) = 50;
    N_points(((i-1)*8+26)*n_sim_scen+2:((i-1)*8+27)*n_sim_scen+1) = 100;
    N_points(((i-1)*8+27)*n_sim_scen+2:((i-1)*8+28)*n_sim_scen+1) = 250;
    N_points(((i-1)*8+28)*n_sim_scen+2:((i-1)*8+29)*n_sim_scen+1) = 500;
    N_points(((i-1)*8+29)*n_sim_scen+2:((i-1)*8+30)*n_sim_scen+1) = 1000;
    N_points(((i-1)*8+30)*n_sim_scen+2:((i-1)*8+31)*n_sim_scen+1) = 2500;
    N_points(((i-1)*8+31)*n_sim_scen+2:((i-1)*8+32)*n_sim_scen+1) = 5000;
end

% OBS: The number of points estimated is a degree of freedom for PF and EnKF

% Memory allocation for output variables
Results = cell(1,3);
Plant = cell(1,3);

for flag_ode = 1:3
    Par = [U0;Par_fix];

    % Memory allocation for system outputs
    Plant{flag_ode}.T = T;
    Plant{flag_ode}.X = zeros(nx,N_k);
    Plant{flag_ode}.Y = zeros(ny,N_k);
    Plant{flag_ode}.Z = zeros(ny,N_k);
    Plant{flag_ode}.Par = repmat(Par,1,N_k);

    % Definition of initial conditions of the simulation
    Plant{flag_ode}.X(1:nx,1) = X0;
    Plant{flag_ode}.Y(1:ny,1) = Y0;
    Plant{flag_ode}.Z(1:ny,1) = Y0;

    % System simulation under process and measurements noises
    for k = 1:N_k-1
        % Adding disturbances to the system
        if k == 56
            Par(1:2) = [114.144476079745; -2825.08863998444];
%             Par(1) = [114.144476079745];
            % OBS: The previous values refer to a steady-state found for C_b = 0.7
        end
        if k == 81+15
            Par(1:2) = [216.779005309737; -5081.48878318955];
%             Par(2) = [-5081.48878318955];
            % OBS: The previous values refer to a steady-state found for C_b = 0.95
        end
        if k == 16 && flag_ode>1
            Par(3) = 4.5;
        end

        % Adding process noises through model parameters
        Plant{flag_ode}.Par(1:npar,k) = Par + normrnd(0,w_U);

        % System update
        Plant{flag_ode}.X(1:nx,k+1) = F_real(Plant{flag_ode}.X(1:nx,k), Plant{flag_ode}.Par(1:npar,k));
        Plant{flag_ode}.Y(1:ny,k+1) = H(Plant{flag_ode}.X(1:nx,k+1));
        Plant{flag_ode}.Z(1:ny,k+1) = Plant{flag_ode}.Y(1:ny,k+1) + normrnd(0,vk);
    end

    % Parameter estimated and model inputs based on their position in "Par"
    A_par = eye(npar);
    if flag_ode == 3
        pos_var = 3;
        pos_u = 1:2;
    else
        pos_var = [];
        pos_u = 1:3;
    end
    npar_var = numel(pos_var);
    nest = nx+npar_var;
    nu = numel(pos_u);

    % Definition of matrices to define the state transition function for the observer
    A_par_var = zeros(npar,nest);
    for i = 1:npar_var
        A_par(pos_var(i),pos_var(i)) = 0;
        A_par_var(pos_var(i),nx+i) = 1;
    end
    A_par_u = zeros(npar,nu);
    for i = 1:nu
        A_par(pos_u(i),pos_u(i)) = 0;
        A_par_u(pos_u(i),i) = 1;
    end
    A_x_var = zeros(nx,nest);
    for i = 1:nx
        A_x_var(i,i) = 1;
    end

    % Definition of the augmented state
    Plant{flag_ode}.X = [Plant{flag_ode}.X;Plant{flag_ode}.Par(pos_var,:)];

    % Definition of the state transition function for the observer
    F = @(var,U) [solvingODE(@(t,x,u) ODE_VanderVusse(t,x,u), Ts,...
        A_x_var*var, A_par*Par+A_par_var*var+A_par_u*U, optsODE);var(nx+1:nest)];

%     F = @(var,U) solvingODE(@(t,x,u) ODE_VanderVusse(t,x,u), Ts,...
%         A_x_var*var, A_par*Par+A_par_var*var+A_par_u*U, optsODE);

    % Definition of state transition matrix and measurement matrix for the observer
    Jacob_x = @(var,U) expm(Jacobian_VanderVusse(var,A_par*Par+A_par_var*var+A_par_u*U)*Ts);
    Jacob_y = zeros(ny, nest);
    Jacob_y(1:ny,1:nx) = Jacob_y0;

    % Definition of covariances matrices of process (Q) and observation (R) noises
    w_U(pos_var) = 0.01; % Artificial noise considered for flag_ode = 3
    dQ = [dF_dPar;zeros(npar_var,npar)];
    for i = 1:npar_var
        dQ(nx+i,pos_var(i)) = 1;
    end
    Q = dQ*(diag(w_U).^2)*dQ';
    R = diag(vk.^2);

    % Memory allocation for estimation outputs
    Results{flag_ode}.X_sim = zeros(nest,N_k,n_sim);
    Results{flag_ode}.Y_sim = zeros(ny,N_k,n_sim);
    Results{flag_ode}.Pxx_sim = zeros(nest,nest,N_k,n_sim);
    Results{flag_ode}.X_vector = zeros(nest,N_k,n_sim);
    Results{flag_ode}.Pxx_vector = zeros(nest,nest,N_k,n_sim);
    Results{flag_ode}.Esf_vector = zeros(1,N_k,n_sim);

% Lembrar definir os calculos em funçao de um Par0

    % Simulation
    for i = 1:n_sim
        [Results{flag_ode}.X_sim(1:nest, 1:N_k,i), Results{flag_ode}.Y_sim(1:ny, 1:N_k,i),...
            Results{flag_ode}.Pxx_sim(1:nest, 1:nest, 1:N_k,i),...
            Results{flag_ode}.X_vector(1:nest, 1:N_k,i),...
            Results{flag_ode}.Pxx_vector(1:nest, 1:nest, 1:N_k,i),...
            Results{flag_ode}.Esf_vector(1,1:N_k,i)] = Estimation_Unconst(F,...
            H, [X0;Par0(pos_var)], Plant{flag_ode}.Par(pos_u,:), Plant{flag_ode}.Z,...
            Q, R, Jacob_x, Jacob_y, N_smo(i), flag_est(i), N_points(i));
    end
    Results{flag_ode}.Q = Q;
    Results{flag_ode}.R = R;
end

[Results, Table] = Plotting_Est(Plant, Results, n_sim_scen, N_smo, N_points, dir_save);

save(strcat(dir_save,"Unconst.mat"),'Plant','Results', 'n_sim_scen', 'N_smo', 'N_points', 'Table');