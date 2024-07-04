% Main script of a comparative study of constrained forward and forward-backward
% Bayesian filters based on a for a Van de Vusse reactor"
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
% Constraints - Structure comprising the set of model constraints
% Constraints.g - Nonlinear inequality constraints in the form g(x) <= 0
% Constraints.h - Nonlinear equality constraints in the form h(x) = 0
% Constraints.A - Linear coefficient of inequality constraints in the form A*x <= B
% Constraints.B - Constant of inequality constraints in the form A*x <= B
% Constraints.Aeq - Linear coefficient of equality constraints in the form Aeq*x <= Beq
% Constraints.Beq - Constant of equality constraints in the form Aeq*x <= Beq
% Constraints.lbx - Lower boundary of the feasible region in the form x \in [lbx,ubx]
% Constraints.ubx - Upper boundary of the feasible region in the form x \in [lbx,ubx]
%
% References:
% Salau, N. P. G., Trierweiler, J. O. & Secchi, A. R. State estimators for
% better bioprocesses operation. Comput. Aided Chem. Eng. 30, 1267–1271,
% doi: 10.1016/B978-0-444-59520-1.50112-3 (2012).
%
% van de Vusse, J. G. (1964) Plug-flow type reactor versus
% tank reactor. Chem. Engng Sci. 19(6), 994--998
%
% Trierweiler, J. O., & Diehl, F. C. (2009). Analysis, control, and operational
% optimization of a Zymomonas mobilis reactor with equilibrium multiplicity.
% IFAC Proceedings Volumes, 42(11), 159-164.
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024

clear all;
close all;

% Definition of ODE solver configurations
Jpat = sparse([true	    false	true	false;...
    true	    true	true	false;...
    true	    true	true	true;...
    false	false	true	true]);

% Definition of the data directory
dir_save = "..\..\Results\Salau\";
if ~exist(dir_save,'dir')
    mkdir(dir_save);
end
optsODE = odeset( 'Vectorized', 'off', 'JPattern', []);

% Definition of model parameters
Par_fix = [0; 0; 0; 0.00383; 59.2085; 70.5565; 0.500; 2.160; 1.100; 0.02445;...
    0.05263; 1.0];

% Definition of the observer for each simulated scenarios
n_scen = 8*6; % Number of different scenarios
n_sim_scen = 1; % Number of simulations for each scenario
n_sim = 1+n_sim_scen*n_scen; % Total number of simulations
% OBS: First simulation is discarded to compensate the just-in-time compiler

N_smo = zeros(1,n_sim); % Smoothing or moving horizon
flag_est = zeros(1,n_sim); % Definition of the observer algorithm
N_points = zeros(1,n_sim); % Number of particles or ensemble members,
% depending on the observer
flag_est(1:8*n_sim_scen+1) = 1; % CEKF&S
flag_est(8*n_sim_scen+2:16*n_sim_scen+1) = 2; % CUKF&S
flag_est(16*n_sim_scen+2:24*n_sim_scen+1) = 3; % CCKF&S
flag_est(24*n_sim_scen+2:32*n_sim_scen+1) = 4; % PF&S
flag_est(32*n_sim_scen+2:40*n_sim_scen+1) = 5; % CEnKF&S
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
% OBS: Smoothing horizon is defined only for CEKF&S, CUKF&S, and CCKF&S

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

    % Definition of the initial plant condition based on the standard condition
    % reported by Salau et al. (2012)
    X0 = [1.24084109679516; 4.73481889735225; 13.3112641695954; 92.5633585753498];
    U0 = [2.00388580231628; 199.997654179592];

    % Definition of the set of time-invariant model parameters
    Par = [U0;Par_fix];

    % Definition of manipulated variables based on their position in Par
    pos_u = [1:2,5];

    % Definition of initial condition based on the standard condition reported
    % by Salau et al. (2012)
    if flag_ode < 2
        pos_u = [1:2];

        % Sampling time [h]
        Ts = 1/60;

        % Analysis period [h]
        T = 0:Ts:10;

        % Measurement function
        H = @(x0) x0(1);

        % Parameter estimated based on their position in "Par"
        pos_var = [];

        % Covariance matrix of observation noises
        R = 1^2;

        % Covariance matrix of process noises
        Q = 0.01^2*eye(4);

        % Initial state error covariance matrix
        P0 = 0.05^2*eye(4);

    else
        pos_u = [1:2,5];
        % Sampling time [h]
        Ts = 0.25;

        % Analysis period [h]
        T = 0:Ts:24;

        % Measurement function
        H = @(var) [var(1);var(4)];

        % Covariance matrix of observation noises
        R = 0.1^2*eye(2);

        if flag_ode > 2
            % Parameter estimated based on their position in "Par"
            pos_var = 1:2;

            % Covariance matrix of process noises
            Q = diag([0.05^2 0.05^2 0.05^2 0.05^2 0.1^2 0.1^2]);

            % Initial state error covariance matrix
            P0 = 0.75^2*eye(6);
        else
            % Parameter estimated based on their position in "Par"
            pos_var = 7;

            % Covariance matrix of process noises
            Q = 0.5^2*eye(5);

            % Initial state error covariance matrix
            P0 = 0.05^2*eye(5);
        end
    end
    Y0 = H(X0);

    % Definition of the number of states, inputs, model parameters, and measurements
    nx = numel(X0);
    nu = numel(pos_u);
    npar = numel(Par);
    ny = numel(Y0);

    % Definition of the number of augmented states and estimated parameters
    npar_var = numel(pos_var);
    nest = nx+npar_var;

    % Definition of matrices to define the state transition function for the observer
    A_par = eye(npar);
    A_par_u = zeros(npar,nu);
    for i = 1:nu
        A_par(pos_u(i),pos_u(i)) = 0;
        A_par_u(pos_u(i),i) = 1;
    end
    A_par_var = zeros(npar,nest);
    for i = 1:npar_var
        A_par(pos_var(i),pos_var(i)) = 0;
        A_par_var(pos_var(i),nx+i) = 1;
    end
    A_x_var = zeros(nx,nest);
    for i = 1:nx
        A_x_var(i,i) = 1;
    end

    vk = diag(R.^0.5);
    wk = diag(Q.^0.5);

    % Definition of the state transition and measurement functions
    F_real = @(var,Par) solvingODE(@(t,x,u) ODE_JobsesZymomonas(t,x,u), Ts,...
        var, Par, optsODE);

    % Number of sampling times in each simulation
    N_k = numel(T);

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

    y_sp = Plant{flag_ode}.X([4,1],1);
    cumerror = zeros(2,1);
    % System simulation based on the scenarios proposed by Salau et al. (2012)
    for k = 1:N_k-1
        % Adding disturbances to the system
        if flag_ode > 1
            if k == 21
                Par(1) = 2.5;
            end
            if k == 41
                Par(1) = U0(1);
            end
            if k == 61
                Par(5) = 81;
            end
            if k == 67
                Par(5) = 0;
            end
            du = zeros(nu,1);
        else
            if k == 61
                y_sp = [96; 1];
            end
            if k == 121
                y_sp = [90; 1.5];
            end
            if k == 181
                y_sp = [94; 1.24];
            end
            if k == 241
                y_sp = [98; 1.5];
            end
            if k == 301
                y_sp = [98; 1.0];
            end
            if k == 361
                y_sp = [98; 0.5];
            end
            if k == 421
                y_sp = [100; 0.5];
            end
            if k == 481
                y_sp = [100; 1.0];
            end
            if k == 541
                y_sp = [97; 1.5];
            end
            cumerror = cumerror + (y_sp - Plant{flag_ode}.X([4,1],k))*Ts;
            Kp = 0.4*[-0.016,6.133;0.7077,3.2102];
            Ki = [-0.016/0.565,6.133/0.8076;0.5*0.7077/0.015,3.2102/0.4393];
            du = Kp'*(y_sp - Plant{flag_ode}.X([4,1],k)) + Ki'*cumerror;
        end
        Plant{flag_ode}.Par(1:npar,k) = Par;
        Plant{flag_ode}.Par(1:nu,k) = Plant{flag_ode}.Par(1:nu,k) + du;

        % System update
        Plant{flag_ode}.X(1:nx,k+1) = F_real(Plant{flag_ode}.X(1:nx,k),...
            Plant{flag_ode}.Par(1:npar,k)) + normrnd(0,wk((1:nx)));
        Plant{flag_ode}.Y(1:ny,k+1) = H(Plant{flag_ode}.X(1:nx,k+1));
        Plant{flag_ode}.Z(1:ny,k+1) = Plant{flag_ode}.Y(1:ny,k+1) + normrnd(0,vk);
    end

    % Definition of the state transition function for the observer
    F = @(var,U) [solvingODE(@(t,x,u) ODE_JobsesZymomonas(t,x,u), Ts,...
        A_x_var*var, A_par*Par+A_par_var*var+A_par_u*U, optsODE);var(nx+1:nest)];

    % Definition of all model constraints within a single struct
    Constraints = struct;
    % Definition of nonlinear inequality constraints g(x) <= 0
    Constraints.g = @(var,u) [];
    % Definition of nonlinear equality constraints h(x) = 0
    Constraints.h = @(var,u) [];
    % Definition of linear inequality constraints in the form A*x <= B
    Constraints.A = @(var,u) [];
    Constraints.B = @(var,u) [];
    % Definition of linear equality constraints in the form Aeq*x = Beq
    Constraints.Aeq = @(var,u) [];
    Constraints.Beq = @(var,u) [];
    % Definition of boundary constraints x \in [lbx,ubx]
    switch flag_ode
        case 1
            Constraints.lbx = [0.01; 0.01; 0.01; 80];
            Constraints.ubx = [10;20;50;110];
        case 2
            Constraints.lbx = [0.15; 1.2; 1.8; 30; 53.28];
            Constraints.ubx = [150; 5; 41; 121; 65.13];
        case 3
            Constraints.lbx = [0.15; 1.2; 1.8; 30; -5; 0];
            Constraints.ubx = [150; 5; 41; 121; 0; 0];
    end

    % Definition of state transition matrix and measurement matrix for the observer
    Jacob_x = @(var,U) expm(Jacobian_JobsesZymomonas(var,A_par*Par+A_par_var*var+A_par_u*U)*Ts);
    if flag_ode == 1
        Jacob_y = zeros(1,nest);
        Jacob_y(1) = 1;
    else
        Jacob_y = zeros(2,nest);
        Jacob_y(1,1) = 1;
        Jacob_y(2,4) = 1;
    end

    % Memory allocation for estimation outputs
    Results{flag_ode}.X_sim = zeros(nest,N_k,n_sim);
    Results{flag_ode}.Y_sim = zeros(ny,N_k,n_sim);
    Results{flag_ode}.PxX_sim = zeros(nest,nest,N_k,n_sim);
    Results{flag_ode}.X_vector = zeros(nest,N_k,n_sim);
    Results{flag_ode}.Pxx_vector = zeros(nest,nest,N_k,n_sim);
    Results{flag_ode}.Esf_vector = zeros(1,N_k,n_sim);

    % Definition of initial observer guesses based on the simulations from Salau et al. (2012)
    if flag_ode == 2
        X0 = [8.77998822517020;4.55394796449275;9.62863017422396;89.0508545283520];
        U0 = [2.00388580231628; 199.997654179592];
        Par(pos_var) = 56.25;
    end
    if flag_ode == 3
        X0 = [111.339988410917;2.11159556862693;4.24136214105304;41.2903415223049];
        U0 = [2.00388580231628; 199.997654179592];
        Par(1:2) = 0;
    end

    % Definition of the augmented state
    Plant{flag_ode}.X = [Plant{flag_ode}.X; Plant{flag_ode}.Par(pos_var,:)];
    Plant{flag_ode}.U = Plant{flag_ode}.Par(pos_u,:);
    if flag_ode == 3
        Plant{flag_ode}.U(1,16:N_k) = Plant{flag_ode}.U(1,16:N_k)+2;
        Plant{flag_ode}.X(nx+1,1:15) = 0;
        Plant{flag_ode}.X(nx+1,16:N_k) = -2;
        Plant{flag_ode}.X(nx+2,:) = 0;
    end

    % Simulation
    for i = 1:n_sim
        [Results{flag_ode}.X_sim(1:nest, 1:N_k,i), Results{flag_ode}.Y_sim(1:ny, 1:N_k,i),...
            Results{flag_ode}.Pxx_sim(1:nest, 1:nest, 1:N_k,i), Results{flag_ode}.X_vector(1:nest, 1:N_k,i),...
            Results{flag_ode}.Pxx_vector(1:nest, 1:nest, 1:N_k,i), Results{flag_ode}.Esf_vector(1, 1:N_k,i)] =...
            Estimation_Const(F, H, [X0;Par(pos_var)], Plant{flag_ode}.U, Plant{flag_ode}.Z,...
            Q, R, Jacob_x, Jacob_y, Constraints, N_smo(i), flag_est(i), N_points(i),Plant{flag_ode}.X, P0);
    end
    Results{flag_ode}.Q = Q;
    Results{flag_ode}.R = R;
end

%[Results, Table] = Plotting_Est(Plant, Results, n_sim_scen, N_smo, N_points, dir_save);
save(strcat(dir_save,"Salau.mat"),'Plant','Results', 'n_sim_scen', 'N_smo', 'N_points');