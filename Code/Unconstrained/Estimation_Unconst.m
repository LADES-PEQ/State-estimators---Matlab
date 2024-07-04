function [X_est, Y_est, Pxx_est, X_vector, Pxx_vector, Esf_vector] =...
    Estimation_Unconst(F, H, X0, U_vector, Z_vector, Q, R, Jacob_x, Jacob_y,...
    N_smo, flag_est, N_points, varargin)
% Recursive estimation of an unconstrained system for a predefined analysis
% period based on a Bayesian filter defined by flag_est
%
% Inputs:
% F - State transition function
% H - Measurement function
% X0 - State at the beginning of the analysis period
% U_vector - Set of time-invariant variables within the analysis period
% Z_vector - Measurements from the system at time k
% Q - Covariance matrix of the process noises
% R - Covariance matrix of the observation noises
% Jacob_x - Numerical calculation of the state transition matrix
% Jacob_y - Measurement matrix
% N_smo - Smoothing horizon
% flag_est - Indicator of the observer algorithm
% N_points - Number of state points calculated at each sampling time
%
% Outputs:
% X_est - Real-time state estimations within the analysis period
% Y_est - Real-time measurement predictions within the analysis period
% Pxx_est - Real-time state error covariance matrices within the analysis period
% X_vector - Moving horizon estimations within the analysis period
% Pxx_vector - Moving horizon state error covariance matrices within the analysis period
% Esf_vector - Observer execution time within the analysis period
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The variables with the prefix "Points" have calculations for each
% points propagated based on the column position.
%
% Note2: The execution time of the forward-backward filtering is extremely
% slow for a large number of particles. In addition, preliminary tests
% showed that estimation accuracy is more sensitive to particle size than
% the smoothing horizon. Hence, this algorithm is mainly intended for severe
% fine-tuning of the observer.

% Definition of the number of states, measurements, sampling times and
% model input parameters
nx = numel(X0);
[ny,N_k] = size(Z_vector);
nu = size(U_vector,1);

% Calculation of the state error covariance matrix at the beginning of the
% analysis period (P0) based on the simulation starting at steady state
phi = Jacob_x(X0,U_vector(1:nu,1)); % State transition matrix
P0 = idare(phi',Jacob_y',Q,R,zeros(nx,ny), eye(nx));

% Memory allocation for output variables of the simulation
X_est = zeros(nx,N_k);
Y_est = zeros(ny,N_k);
Pxx_est = zeros(nx,nx,N_k);
X_vector = zeros(nx,N_k);
Pxx_vector = zeros(nx,nx,N_k);
Esf_vector = zeros(N_k,1);

% Definition of initial conditions of the simulation
Q_vector = repmat(Q,1,1,N_k);
R_vector = repmat(R,1,1,N_k);
X_vector(1:nx,1) = X0;
X_est(1:nx,1) = X0;
Y_est(1:ny,1) = H(X0);
Pxx_vector(1:nx,1:nx,1) = P0;
Pxx_est(1:nx,1:nx,1) = P0;

% Memory allocation for other variables recursively calculated in the simulation
if flag_est == 2
    N_points = 2*nx+1;
end
if flag_est == 3
    N_points = 2*nx;
end
if flag_est <= 1
    Xpri_vector = zeros(nx,N_k);
    phi_vector =  zeros(nx,nx,N_k);
else
    Points_vector = zeros(nx,N_points,N_k);
    Points_priori = zeros(nx,N_points,N_k);
end
if flag_est > 3
    Q_chol = chol(Q,"lower");
    for i = 1:N_points
        Points_vector(1:nx,i,1) = X0 + Q_chol*randn(nx,1);
    end
    if flag_est == 4
        Particle_W = zeros(N_points,N_k);
        Particle_W(1:N_points,1) = 1/N_points;
    end
end

N_max = N_smo;
N_smo = 0;
% Recursive estimations
for k = 1:N_k-1
%     if k == N_max+1
%         N_smo = N_max;
%     end
    N_smo = min(N_max,k-1);

    tic
    % Observer definition based on flag_est
    switch flag_est
        case 0 % Steady extended Kalman filter and smoother (SEKFS)
            [Pxx_vector(1:nx,1:nx,k-N_smo:k+1), X_vector(1:nx,k-N_smo:k+1),...
                Xpri_vector(1:nx,k-N_smo:k+1), phi_vector(1:nx,1:nx,k-N_smo:k+1)] =...
                EKFS(F, H, X_vector(1:nx,k-N_smo:k), U_vector(1:nu,k-N_smo:k),...
                Pxx_vector(1:nx,1:nx,k-N_smo:k), Z_vector(1:ny,k-N_smo:k+1),...
                Q_vector(1:nx,1:nx,k-N_smo:k), R_vector(1:ny,1:ny,k-N_smo:k+1),...
                Jacob_x, Jacob_y, Xpri_vector(1:nx,k-N_smo:k), phi_vector(1:nx,1:nx,k-N_smo:k), 1);

        case 1 % Extended Kalman filter and smoother (EKFS)
            [Pxx_vector(1:nx,1:nx,k-N_smo:k+1), X_vector(1:nx,k-N_smo:k+1),...
                Xpri_vector(1:nx,k-N_smo:k+1), phi_vector(1:nx,1:nx,k-N_smo:k+1)] =...
                EKFS(F, H, X_vector(1:nx,k-N_smo:k), U_vector(1:nu,k-N_smo:k),...
                Pxx_vector(1:nx,1:nx,k-N_smo:k), Z_vector(1:ny,k-N_smo:k+1),...
                Q_vector(1:nx,1:nx,k-N_smo:k), R_vector(1:ny,1:ny,k-N_smo:k+1),...
                Jacob_x, Jacob_y, Xpri_vector(1:nx,k-N_smo:k), phi_vector(1:nx,1:nx,k-N_smo:k), 0);

        case 2 % Unscented Kalman filter and smoother (UKFS)
            [Pxx_vector(1:nx,1:nx,k-N_smo:k+1), X_vector(1:nx,k-N_smo:k+1),...
                Points_vector(1:nx,1:2*nx+1,k-N_smo:k+1),...
                Points_priori(1:nx,1:2*nx+1,k-N_smo:k+1)] = UKFS(F, H,...
                X_vector(1:nx,k-N_smo:k), U_vector(1:nu,k-N_smo:k),...
                Pxx_vector(1:nx,1:nx,k-N_smo:k), Z_vector(1:ny,k-N_smo:k+1),...
                Q_vector(1:nx,1:nx,k-N_smo:k), R_vector(1:ny,1:ny,k-N_smo:k+1),...
                Points_vector(1:nx,1:2*nx+1,k-N_smo:k), Points_priori(1:nx,1:2*nx+1,k-N_smo:k));

        case 3 % Cubature Kalman filter and smoother (CKFS)
            [Pxx_vector(1:nx,1:nx,k-N_smo:k+1), X_vector(1:nx,k-N_smo:k+1),...
                Points_vector(1:nx,1:2*nx,k-N_smo:k+1),...
                Points_priori(1:nx,1:2*nx,k-N_smo:k+1)] = CKFS(F, H,...
                X_vector(1:nx,k-N_smo:k), U_vector(1:nu,k-N_smo:k),...
                Pxx_vector(1:nx,1:nx,k-N_smo:k), Z_vector(1:ny,k-N_smo:k+1),...
                Q_vector(1:nx,1:nx,k-N_smo:k), R_vector(1:ny,1:ny,k-N_smo:k+1),...
                Points_vector(1:nx,1:2*nx,k-N_smo:k), Points_priori(1:nx,1:2*nx,k-N_smo:k));

        case 4 % Particle filter and smoother (PFS)
            [Pxx_vector(1:nx,1:nx,k-N_smo:k+1), X_vector(1:nx,k-N_smo:k+1),...
                Points_vector(1:nx,1:N_points,k-N_smo:k+1),...
                Particle_W(1:N_points,k-N_smo:k+1), Points_priori(1:nx,1:N_points,k-N_smo:k+1)] =...
                PFS(F, H, Points_vector(1:nx,1:N_points,k-N_smo:k), U_vector(1:nu,k-N_smo:k),...
                Z_vector(1:ny,k-N_smo:k+1), Q_vector(1:nx,1:nx,k-N_smo:k),...
                R_vector(1:ny,1:ny,k-N_smo:k+1), Particle_W(1:N_points,k-N_smo:k),...
                Points_priori(1:nx,1:N_points,k-N_smo:k));
            
        case 5 % Ensemble Kalman filter and smoother (EnKFS)
            [Pxx_vector(1:nx,1:nx,k-N_smo:k+1), X_vector(1:nx,k-N_smo:k+1),...
                Points_vector(1:nx,1:N_points,k-N_smo:k+1),...
                Points_priori(1:nx,1:N_points,k-N_smo:k+1)] = EnKFS(F, H,...
                Points_vector(1:nx,1:N_points,k-N_smo:k), U_vector(1:nu,k-N_smo:k),...
                Pxx_vector(1:nx,1:nx,k-N_smo:k), Z_vector(1:ny,k-N_smo:k+1),...
                Q_vector(1:nx,1:nx,k-N_smo:k), R_vector(1:ny,1:ny,k-N_smo:k+1),...
                Jacob_y, X_vector(1:nx,k-N_smo:k),...
                Points_priori(1:nx,1:N_points,k-N_smo:k));
    end

    Esf_vector(k+1) = toc;
    X_est(1:nx,k+1) = X_vector(1:nx,k+1);
    Y_est(1:ny,k+1) = H(X_vector(1:nx,k+1));
    Pxx_est(1:nx,1:nx,k+1) = Pxx_vector(1:nx,1:nx,k+1);

end
end