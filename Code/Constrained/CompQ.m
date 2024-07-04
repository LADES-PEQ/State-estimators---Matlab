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
dir_save = "..\..\Resultados\";
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
flag_est(24*n_sim_scen+2:32*n_sim_scen+1) = 0*4; % PF&S
flag_est(32*n_sim_scen+2:40*n_sim_scen+1) = 0*5; % CEnKF&S
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

for flag_ode = 3:3
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
    pos_u = [1:2,5];

    % Sampling time [h]
    Ts = 0.25;

    % Analysis period [h]
    T = 0:Ts:24;

    % Measurement function
    H = @(var) [var(1);var(4)];

    % Covariance matrix of observation noises
    R = 0.1^2*eye(2);
    if flag_ode < 3
        % Parameter estimated based on their position in "Par"
        pos_var = [];

        % Covariance matrix of process noises
        Q = 0.05^2*eye(4);

    else
        pos_var = 1:2;
        % Covariance matrix of process noises
        Q = diag([0.05^2 0.05^2 0.05^2 0.05^2 0.1^2 0.1^2]);
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
        %         A_par_u(pos_var(i),:) = 0;
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

    cumerror = zeros(2,1);
    % System simulation based on the scenarios proposed by Salau et al. (2012)
    for k = 1:N_k-1
        % Adding disturbances to the system
        if k == 31
            Par(1) = 2.5;
        end
        if k == 51
            Par(1) = U0(1);
        end
        if k == 64
            Par(5) = 81;
        end
        if k == 70
            Par(5) = 0;
        end

        Plant{flag_ode}.Par(1:npar,k) = Par;
        Plant{flag_ode}.Par(1:nu,k) = Plant{flag_ode}.Par(1:nu,k);


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
    Constraints.g = [];
    % Definition of nonlinear equality constraints h(x) = 0
    Constraints.h = [];
    % Definition of linear inequality constraints in the form A*x <= B
    Constraints.A = [];
    Constraints.B = [];
    % Definition of linear equality constraints in the form Aeq*x = Beq
    Constraints.Aeq = [];
    Constraints.Beq = [];
    % Definition of boundary constraints x \in [lbx,ubx]
    switch flag_ode
        case 1
            Constraints.lbx = [0.15; 1.2; 1.8; 30];
            Constraints.ubx = [150; 5; 41; 121];
        case 2
            Constraints.lbx = [0.15; 1.2; 1.8; 30];
            Constraints.ubx = [150; 5; 41; 121];
        case 3
            Constraints.lbx = [0.15; 1.2; 1.8; 30; -(2+U0(1)); 0];
            Constraints.ubx = [150; 5; 41; 121; 0; 0];
    end
    %     Constraints.lbx = [];
    %     Constraints.ubx = [];

    % Definition of state transition matrix and measurement matrix for the observer
    Jacob_x = @(var,U) expm(Jacobian_JobsesZymomonas(var,A_par*Par+A_par_var*var+A_par_u*U)*Ts);
    Jacob_y = zeros(2,nest);
    Jacob_y(1,1) = 1;
    Jacob_y(2,4) = 1;

    % Memory allocation for estimation outputs
    Results2{flag_ode}.X_sim = zeros(nest,N_k,n_sim);
    Results2{flag_ode}.Y_sim = zeros(ny,N_k,n_sim);
    Results2{flag_ode}.Pxx_sim = zeros(nest,nest,N_k,n_sim);
    Results2{flag_ode}.X_vector = zeros(nest,N_k,n_sim);
    Results2{flag_ode}.Pxx_vector = zeros(nest,nest,N_k,n_sim);
    Results2{flag_ode}.Esf_vector = zeros(1,N_k,n_sim);


    % Definition of initial observer guesses based on the simulations from Salau et al. (2012)
    if flag_ode == 2
        %         X0 = [8.77998822517020;4.55394796449275;9.62863017422396;89.0508545283520];
        %         U0 = [2.00388580231628; 199.997654179592];

        %         Par(pos_var) = 56.25;
    end
    if flag_ode >1
        %         X0 = [111.339988410917;2.11159556862693;4.24136214105304;41.2903415223049];
        %         U0 = [2.00388580231628; 199.997654179592];
        %         Plant{flag_ode}.Par(pos_u,:) = repmat(Par(pos_u)-[2;200],1,N_k);
        Par(1:2) = 0;
    end

    %     U_ref(1) = min(U_ref(1),2);
    %     phi = Jacob_x(X0,U_ref); % State transition matrix
    %     P0 = idare(phi',Jacob_y',Q,R,zeros(nx,ny), eye(nx));

    % Definition of the augmented state
    Plant{flag_ode}.X = [Plant{flag_ode}.X; Plant{flag_ode}.Par(pos_var,:)];
    Plant{flag_ode}.U = Plant{flag_ode}.Par(pos_u,:);
    if flag_ode > 1
        Plant{flag_ode}.U(1,16:N_k) = Plant{flag_ode}.U(1,16:N_k)+2;
    end
    if flag_ode == 3


        Plant{flag_ode}.X(nx+1,1:15) = 0;
        Plant{flag_ode}.X(nx+1,16:N_k) = -2;
        Plant{flag_ode}.X(nx+2,:) = 0;
    end

    load(strcat(dir_save,"Const.mat"));

    n_sim_scen = 1;
    Q_var =  zeros(1,n_sim);
    Q_var(1:8*n_sim_scen+1) = 1e-6;
    Q_var(8*n_sim_scen+2:16*n_sim_scen+1) = 1e-5;
    Q_var(16*n_sim_scen+2:24*n_sim_scen+1) = 1e-4;
    Q_var(24*n_sim_scen+2:32*n_sim_scen+1) = 1e-2;
    Q_var(32*n_sim_scen+2:40*n_sim_scen+1) = 1e-1;
    Q_var(40*n_sim_scen+2:48*n_sim_scen+1) = 1;

    flag_est(1:8*n_sim_scen+1) = 2; % CEKF&S
    flag_est(8*n_sim_scen+2:16*n_sim_scen+1) = 2; % CUKF&S
    flag_est(16*n_sim_scen+2:24*n_sim_scen+1) = 2; % CCKF&S
    flag_est(24*n_sim_scen+2:32*n_sim_scen+1) = 2; % PF&S
    flag_est(32*n_sim_scen+2:40*n_sim_scen+1) = 2; % CEnKF&S
    flag_est(40*n_sim_scen+2:48*n_sim_scen+1) = 2; % CEnKF&S
    for i = 1:6
        N_smo(((i-1)*8+0)*n_sim_scen+2:((i-1)*8+1)*n_sim_scen+1) = 0;
        N_smo(((i-1)*8+1)*n_sim_scen+2:((i-1)*8+2)*n_sim_scen+1) = 1;
        N_smo(((i-1)*8+2)*n_sim_scen+2:((i-1)*8+3)*n_sim_scen+1) = 2;
        N_smo(((i-1)*8+3)*n_sim_scen+2:((i-1)*8+4)*n_sim_scen+1) = 4;
        N_smo(((i-1)*8+4)*n_sim_scen+2:((i-1)*8+5)*n_sim_scen+1) = 6;
        N_smo(((i-1)*8+5)*n_sim_scen+2:((i-1)*8+6)*n_sim_scen+1) = 8;
        N_smo(((i-1)*8+6)*n_sim_scen+2:((i-1)*8+7)*n_sim_scen+1) = 10;
        N_smo(((i-1)*8+7)*n_sim_scen+2:((i-1)*8+8)*n_sim_scen+1) = 15;
    end
    % Simulation
    for i = 1:n_sim
        Q(5,5) = Q_var(i);
        Q(6,6) = Q_var(i);


        [Results2{flag_ode}.X_sim(1:nest, 1:N_k,i), Results2{flag_ode}.Y_sim(1:ny, 1:N_k,i),...
            Results2{flag_ode}.Pxx_sim(1:nest, 1:nest, 1:N_k,i), Results2{flag_ode}.X_vector(1:nest, 1:N_k,i),...
            Results2{flag_ode}.Pxx_vector(1:nest, 1:nest, 1:N_k,i), ~] =...
            Estimation_Const(F, H, [X0;Par(pos_var)], Plant{flag_ode}.U, Plant{flag_ode}.Z,...
            Q, R, Jacob_x, Jacob_y, Constraints, N_smo(i), flag_est(i), N_points(i),Plant{flag_ode}.X);
    end
end


n_case = numel(Results); % Number of case studies

for flag_ode = 3:3
    [~,N_k,n_sim] = size(Results2{flag_ode}.X_sim);
    ny = size(Results2{flag_ode}.Y_sim,1);
    nest = size(Results2{flag_ode}.X_sim,1);

    % Memory allocation of remaining simulation outputs
    Results2{flag_ode}.X_mean = zeros(1,n_scen);
    Results2{flag_ode}.Y_mean = zeros(1,n_scen);
    Results2{flag_ode}.Esf_mean = zeros(1,n_scen);
    Results2{flag_ode}.flag_est = zeros(1,n_scen);
    Results2{flag_ode}.N_smo = zeros(1,n_scen);
    Results2{flag_ode}.N_points = zeros(1,n_scen);
    Results2{flag_ode}.MSE_x = zeros(nest,n_scen);
    Results2{flag_ode}.MAPE_x = zeros(nest,n_scen);
    Results2{flag_ode}.MSE_y = zeros(ny,n_scen);
    Results2{flag_ode}.MAPE_y = zeros(ny,n_scen);
    Results2{flag_ode}.LogLikelihood = zeros(1,n_scen);


    Q(5,5) = Q_var(i);
    % Calculation of inverse matrices
    
    n_scen = (n_sim-1)/n_sim_scen;
    % Calculation and storage of remaining simulation outputs
    for m = 1:n_scen
            Q = Results{flag_ode}.Q;
            Q(5,5) = Q_var((m-1)*n_sim_scen+2);
            Q(6,6) = Q_var((m-1)*n_sim_scen+2);
            inv_Q = mldivide(Q,eye(nest));
            inv_R = mldivide(Results{flag_ode}.R,eye(ny));

            Results2{flag_ode}.flag_est(1,m) = N_smo((m-1)*n_sim_scen+2);
            Results2{flag_ode}.N_smo(1,m) = N_smo((m-1)*n_sim_scen+2);
            Results2{flag_ode}.N_points(1,m) = N_points((m-1)*n_sim_scen+2);
            LogLikelihood = zeros(1,n_sim_scen);
            inv_P0 = inv(Results2{flag_ode}.Pxx_sim(:,:,1,(m-1)*n_sim_scen+2));

            LogLikelihood(1,1) = LogLikelihood(1,1)+(Plant{flag_ode}.Z(1:ny,1)-...
                Results2{flag_ode}.Y_sim(1:ny,1,(m-1)*n_sim_scen+1+1))'*...
                inv_R*(Plant{flag_ode}.Z(1:ny,1)-...
                Results2{flag_ode}.Y_sim(1:ny,1,(m-1)*n_sim_scen+1+1));
            LogLikelihood(1,1) = LogLikelihood(1,1)+(Plant{flag_ode}.X(1:nest,1)-...
                Results2{flag_ode}.X_sim(1:nest,1,(m-1)*n_sim_scen+1+1))'*...
                inv_P0*(Plant{flag_ode}.X(1:nest,1)-Results2{flag_ode}.X_sim(1:nest,1,(m-1)*n_sim_scen+1+1));

            LogLikelihood(1,:) = LogLikelihood(1,1);
            for k = 1:N_k
                if k>1
                    for j = 1:n_sim_scen
                        LogLikelihood(1,j) = LogLikelihood(1,j)+(Plant{flag_ode}.Z(1:ny,k)-...
                            Results2{flag_ode}.Y_sim(1:ny,k,(m-1)*n_sim_scen+1+j))'*...
                            inv_R*(Plant{flag_ode}.Z(1:ny,k)-...
                            Results2{flag_ode}.Y_sim(1:ny,k,(m-1)*n_sim_scen+1+j));
                        LogLikelihood(1,j) = LogLikelihood(1,j)+(Plant{flag_ode}.X(1:nest,k)-...
                            Results2{flag_ode}.X_sim(1:nest,k,(m-1)*n_sim_scen+1+j))'*...
                            inv_Q*(Plant{flag_ode}.X(1:nest,k)-Results2{flag_ode}.X_sim(1:nest,k,(m-1)*n_sim_scen+1+j));
                    end
                end

                for j = 1:nest
                    Results2{flag_ode}.X_mean(j,k,m) = mean(Results2{flag_ode}.X_sim(j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1));
                    Pxx_aux = mean(Results2{flag_ode}.Pxx_sim(j,j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1));
                    Results2{flag_ode}.CI_max_mean(j,k,m) = Results2{flag_ode}.X_mean(j,k,m) + 1.96*Pxx_aux^0.5;
                    Results2{flag_ode}.CI_min_mean(j,k,m) = Results2{flag_ode}.X_mean(j,k,m) - 1.96*Pxx_aux^0.5;
                    Results2{flag_ode}.MSE_x(j,m) = Results2{flag_ode}.MSE_x(j,m) + (1/(n_sim_scen*N_k))*sum(Results2{flag_ode}.X_sim(j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1)-Plant{flag_ode}.X(j,k)).^2;
                    Results2{flag_ode}.MAPE_x(j,m) = Results2{flag_ode}.MAPE_x(j,m) + (1/(n_sim_scen*N_k))*sum(abs((Results2{flag_ode}.X_sim(j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1)-Plant{flag_ode}.X(j,k))/Plant{flag_ode}.X(j,k)));
                end
                for j = 1:ny
                    Results2{flag_ode}.Y_mean(j,k,m) = mean(Results2{flag_ode}.Y_sim(j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1));
                    Results2{flag_ode}.MSE_y(j,m) = Results2{flag_ode}.MSE_y(j,m) + (1/(n_sim_scen*N_k))*sum(Results2{flag_ode}.Y_sim(j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1)-Plant{flag_ode}.Z(j,k)).^2;
                    Results2{flag_ode}.MAPE_y(j,m) = Results2{flag_ode}.MAPE_y(j,m) + (1/(n_sim_scen*N_k))*sum(abs((Results2{flag_ode}.Y_sim(j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1)-Plant{flag_ode}.Z(j,k))/Plant{flag_ode}.Z(j,k)));
                end
            end
            Results2{flag_ode}.LogLikelihood(1,m) = mean(LogLikelihood(1,1:n_sim_scen));
    end
end

Table = [];
for i = 3:3
    aux = Results2{i}.LogLikelihood;

    for j = 1:8
        Table = [Table; [aux(j)/aux(1), aux(8+j)/aux(8+1), aux(16+j)/aux(16+1),...
            aux(24+j)/aux(24+1), aux(32+j)/aux(32+1), aux(40+j)/aux(40+1)]];
    end
end

n_config = 8; % Number of observer configurations evaluated for each simulated scenario
codcolor = [0,0,0;0,130,200;128,128,128;245,130,48;0,0,128;128,0,0;255,225,25;220,190,255;60,180,75]/255;
% Plotting the simulations results under forward-backward filtering
aux = figure;
aux.PaperUnits = 'centimeters';
aux.PaperSize = [13.7 6];
aux.PaperPosition = [0 0 13.7 6];
Margin = 0.02;

    aux2 = subplot(1,1,1);
    bar(0:n_config-1,Table);
    xlabel("Smoothing horizon")
    ylabel('$\overline{J}$', 'Interpreter','latex');
    aux2.Children(6).FaceColor = codcolor(1,:);
    aux2.Children(5).FaceColor = codcolor(2,:);
    aux2.Children(4).FaceColor = codcolor(3,:);
    aux2.Children(3).FaceColor = codcolor(4,:);
    aux2.Children(2).FaceColor = codcolor(5,:);
    aux2.Children(1).FaceColor = codcolor(6,:);

    aux2.Children(6).DisplayName = '$Q^* = 0.000001$';
    aux2.Children(5).DisplayName = '$Q^* = 0.00001$';
    aux2.Children(4).DisplayName = '$Q^* = 0.0001$';
    aux2.Children(3).DisplayName = '$Q^* = 0.01$';
    aux2.Children(2).DisplayName = '$Q^* = 0.1$';
    aux2.Children(1).DisplayName = '$Q^* = 1$';
    aux2.OuterPosition(1) = 0;
    aux2.OuterPosition(2) = 0;
    aux2.OuterPosition(3) = 1;
    aux2.OuterPosition(4) = 1;
    for i = 1:n_config
        aux2.XTickLabel{i} = Results{flag_ode}.N_smo(i);
    end
    aux2.XTickLabelRotation = 0;
    ti_max = aux2.TightInset;
    aux2.Position(1) = ti_max(1);
    aux2.Position(2) = ti_max(2)+0.08;
    aux2.Position(3) = 1-ti_max(1)-ti_max(3)-Margin;
    aux2.Position(4) = 1-aux2.Position(2)-Margin;

    legend('Interpreter','latex')
    aux2.Legend.NumColumns = 2;
    aux2.Legend.FontSize = 6.5;
    aux2.Legend.Location = 'northeast';
    

%     aux2.Legend.Position (1) = aux2.Legend.Position (1) + 0.05;
print("Fig_compQ","-depsc2","-r1000","-vector");

%%
Tagy = ["$C_s \left[\mathrm{kg/m}^3\right]$", "$C_x \left[\mathrm{kg/m}^3\right]$",...
    "$C_e \left[\mathrm{kg/m}^3\right]$", "$C_p \left[\mathrm{kg/m}^3\right]$",...
    "$\Delta D \left[\mathrm{kg/m}^3\right]$", "$\Delta C_{s0} \left[\mathrm{kg/m}^3\right]$"];

codcolor = [0,0,0;0,130,200;128,128,128;245,130,48;0,0,128;128,0,0;255,225,25;220,190,255;60,180,75]/255;
pos_y = [1,4];
aux = figure;
aux.PaperUnits = 'centimeters';
aux.PaperSize = [13.7 12];
aux.PaperPosition = [0 0 13.7 12];
ti_max = 1e5*ones(1,4);
for j = 1:6
    aux2 = subplot(3,2,j);
    plot(Plant{3}.T, Plant{3}.X(j,:), '-', Plant{3}.T, Results2{3}.X_mean(j,:,17-8), '--',...
        Plant{3}.T, Results2{3}.X_mean(j,:,18-8), ':', Plant{3}.T, Results2{3}.X_mean(j,:,20-8), '-.',...
        Plant{3}.T, Results2{3}.X_mean(j,:,22-8), '--', Plant{3}.T, Results2{3}.X_mean(j,:,24-8), '--');

    xlabel('Time (h)', 'Interpreter','latex')
    ylabel(convertStringsToChars(Tagy(j)), 'Interpreter','latex');
    axis tight
    aux2.Children(1).LineStyle = '--';
    aux2.Children(1).Marker = '^';
    aux2.Children(1).MarkerSize = 3;
    aux2.Children(2).LineStyle = '--';
    aux2.Children(2).Marker = 'square';
    aux2.Children(2).MarkerSize = 3;
    aux2.Children(6).LineWidth = 1.5;
    aux2.Children(5).LineWidth = 1.5;
    aux2.Children(4).LineWidth = 1.5;
    aux2.Children(3).LineWidth = 1.5;
    aux2.Children(2).LineWidth = 1.5;
    aux2.Children(1).LineWidth = 1.5;
    aux2.Children(1).Color = codcolor(6,:);
    aux2.Children(2).Color = codcolor(5,:);
    aux2.Children(3).Color = codcolor(4,:);
    aux2.Children(4).Color = codcolor(3,:);
    aux2.Children(5).Color = codcolor(2,:);
    aux2.Children(6).Color = codcolor(1,:);

    aux2.Children(6).DisplayName = "True";
    aux2.Children(5).DisplayName = "CUKF";
    aux2.Children(4).DisplayName = 'CUKFS $\left(N^S = N^*_2\right)$';
    aux2.Children(3).DisplayName = 'CUKFS $\left(N^S = N^*_4\right)$';
    aux2.Children(2).DisplayName = 'CUKFS $\left(N^S = N^*_6\right)$';
    aux2.Children(1).DisplayName = 'CUKFS $\left(N^S = N^*_8\right)$';

    if j == 1
        hold on
        plot(Plant{1}.T, Plant{1}.Z(1,:), 'X');
        aux2.Children(1).LineWidth = 0.5;
        aux2.Children(1).Color = codcolor(1,:);
        aux2.Children(1).DisplayName = "Measurement";
    end
    if j == 4
        hold on
        plot(aux2,Plant{1}.T, Plant{1}.Z(2,:), 'X');
        aux2.Children(1).LineWidth = 0.5;
        aux2.Children(1).Color = codcolor(1,:);
        aux2.Children(1).DisplayName = "Measurement";
    end        
    if j == 6
        aux2.Children(1).Parent.YLim =[-0.1, 0.1];
    end 
    switch j
        case 1
            aux2.OuterPosition(1) = 0;
            aux2.OuterPosition(2) = 2/3;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 1/3;
        case 2
            aux2.OuterPosition(1) = 0.5;
            aux2.OuterPosition(2) = 2/3;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 1/3;
        case 3
            aux2.OuterPosition(1) = 0;
            aux2.OuterPosition(2) = 1/3;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 1/3;
        case 4
            aux2.OuterPosition(1) = 0.5;
            aux2.OuterPosition(2) = 1/3;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 1/3;
        case 5
            aux2.OuterPosition(1) = 0;
            aux2.OuterPosition(2) = 0;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 1/3;
        case 6
            aux2.OuterPosition(1) = 0.5;
            aux2.OuterPosition(2) = 0;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 1/3;
    end
end
ti_max = zeros(1,2);
Margin = 0.01;
for j = 1:6
        aux2 = aux.Children(6+1-j);
    switch j
        case 1
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+1-j-2).Position(1), aux.Children(6+1-j-4).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+1-j-1).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin-0.01;
            aux2.Position(4) = 1-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1)+0.01;
            aux2.Position(2) = ti_max(2);
        case 2
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+1-j-2).Position(1), aux.Children(6+1-j-4).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+1-j+1).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin-0.01;
            aux2.Position(4) = 1-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1)+0.01;
            aux2.Position(2) = ti_max(2);
        case 3
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+1-j+2).Position(1), aux.Children(6+2-j-1).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+1-j-1).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin;
            aux2.Position(4) = 2/3-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
            aux2.NextPlot = 'add';
            plot(aux2, Plant{2}.T, Plant{2}.Z(1,:), 'X');
            aux2.Children(1).LineWidth = 0.5;
            aux2.Children(1).Color = codcolor(1,:);
            aux2.Children(1).DisplayName = "Measurement";
           
            
            legend(aux2.Children([1,7:-1:2]),'Interpreter','latex')
            aux2.Legend.NumColumns = 1;
            aux2.Legend.FontSize = 6.5;
            aux2.Legend.Location = 'best';
            aux2.Legend.AutoUpdate = 'off';
            pause(0.01)
            delete(aux2.Children(1));          
            
        case 4
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+2-j+2).Position(1), aux.Children(6+2-j-1).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+2-j+1).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 2/3-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
        case 5
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+2-j+2).Position(1),aux.Children(6+2-j+4).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+2-j-1).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin;
            aux2.Position(4) = 1/3-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
        case 6
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+2-j+2).Position(1),aux.Children(6+2-j+4).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+2-j).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 1/3-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
            
    end
end
print("FigFinal_CS2","-depsc2","-r1000","-vector");
%%
%13.70499cm

% Results = Plotting_Est(Plant, Results, n_sim_scen, N_smo, N_points, dir_save);
% save(strcat(dir_save,"Const.mat"),'Plant','Results', 'n_sim_scen', 'N_smo', 'N_points');