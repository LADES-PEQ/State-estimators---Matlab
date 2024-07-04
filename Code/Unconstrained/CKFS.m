function [Pxx_filt, x_filt, Cubature_points, Cubature_priori] = CKFS(F, H,...
    X_vector, U_vector, Pxx_vector, z_vector, Q_vector, R_vector,...
    Cubature_points, Cubature_priori, varargin)
% Calculation of the cubature Kalman filter and smoother (CKF&S) for a 
% forward-backward filtering approach, in which the filtering and smoothing
% estimations follow the algorithms proposed by Arasaratnam and Haykin (2009, 2011).
%
% Inputs:
% F - State transition function
% H - Measurement function
% X_vector - State estimations between k-N-1 and k-1
% U_vector - Set of time-invariant variables between k-N-1 and k-1
% Pxx_vector - State error covariance matrices between k-N-1 and k-1
% z_vector - Measurements from the system between k-N-1 and k
% Q_vector - Covariance matrices of the process noises between k-N-1 and k-1
% R_vector - Covariance matrices of the observation noises between k-N-1 and k
% Cubature_points - Set of cubature points defined between k-N-1 and k-1
% Cubature_priori - A priori estimations for each cubature point between k-N and k-1
%
% Outputs:
% Pxx_filt - State error covariance matrices between k-N-1 and k
% x_filt - State estimations between k-N-1 and k
% Cubature_points - Set of cubature points defined between k-N-1 and k-1
% Cubature_priori - A priori estimations for each cubature point between k-N and k
%
% References:
% Arasaratnam, I., & Haykin, S. (2009). Cubature kalman filters. IEEE 
% Transactions on automatic control, 54(6), 1254-1269.
%
% Arasaratnam, I., & Haykin, S. (2011). Cubature kalman smoothers. Automatica,
% 47(10), 2245-2250.
%
% Sarkka, S. (2013). Bayesian filtering and smoothing. Cambridge University Press
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: Equation numbers refer to algorithm transcriptions by Sarkka (2013).
% 
% Note2: The variables with the prefix "Cubature" have calculations for each 
% cubature point propagated based on the column position.

% Definition of the smoothing horizon and the number of states and measurements
[nx,N] = size(X_vector);
ny = size(z_vector,1);
nu = size(U_vector,1);

% Calculation of intermediate parameters
Wm = 1/(2*nx);
Wc = 1/(2*nx);

% Memory allocation for output variables of the smoothing estimations
x_smo = zeros(nx,N);
Pxx_smo = zeros(nx,nx,N);

% Definition of initial conditions for the smoothing estimations based on 
% estimations a posteriori at time k-1
x_smo(1:nx,N) = X_vector(1:nx,N);
Pxx_smo(1:nx,1:nx,N) = Pxx_vector(1:nx,1:nx,N);

% Smoothing estimations from k-2 to k-N-1
for i = 1:N-1
    % Calculation of the state estimations a priori 
    x_priori = zeros(nx,1);
    for j = 1:2*nx
        x_priori = x_priori +Wm*Cubature_priori(1:nx,j,N-i); % Eq. 10.9 part 1
    end

    % Calculation of covariance matrices required to calculate the smoothing gain
    Pxx_priori = Q_vector(1:nx,1:nx,N-i);
    Pxx_cross = zeros(nx,nx);
    for j = 1:2*nx
        Pxx_priori = Pxx_priori+Wc*(Cubature_priori(1:nx,j,N-i)-x_priori)*...
            (Cubature_priori(1:nx,j,N-i)-x_priori)'; % Eq. 10.9 part 2
        Pxx_cross = Pxx_cross+Wc*(Cubature_points(1:nx,j,N-i)-X_vector(1:nx,N-i))*...
            (Cubature_priori(1:nx,j,N-i)-x_priori)'; % Eq. 10.9 part 3
    end

    % Calculation of the smoothing gain
    inv_Ck = mldivide(Pxx_priori,eye(nx));
    inv_Ck = (inv_Ck+inv_Ck')/2;
    Ck = Pxx_cross*inv_Ck; % Eq. 10.10 part 1

    % Estimation of the smoothed counterparts of the states (x_smo), and 
    % the state error covariance matrix (Pxx_smo)
    x_smo(1:nx,N-i) = X_vector(1:nx,N-i) + Ck*(x_smo(1:nx,N-i+1)-x_priori); % Eq. 10.10 part 2
    Pxx_smo(1:nx,1:nx,N-i) = Pxx_vector(1:nx,1:nx,N-i) +...
        Ck*(Pxx_smo(1:nx,1:nx,N-i+1)-Pxx_priori)*Ck'; % Eq. 10.10 part 3
end

% Memory allocation for output variables of the forward filtering pass
x_filt = zeros(nx,N+1);
Pxx_filt = zeros(nx,nx,N+1);

% Definition of the initial condition of the forward filtering pass by the
% smoothed estimation at k-N-1
x_filt(1:nx,1) = x_smo(1:nx,1);
Pxx_filt(1:nx,1:nx,1) = Pxx_smo(1:nx,1:nx,1);
Cubature_points = zeros(nx,2*nx,N+1);
Cubature_priori = zeros(nx,2*nx,N+1);

% Filtering estimations from k-N to k
for i = 1:N
    [Pxx_filt(1:nx,1:nx,i+1), x_filt(1:nx,i+1), Cubature_points(1:nx,1:2*nx,i),...
        Cubature_priori(1:nx,1:2*nx,i)] = CKF(F,H,x_filt(1:nx,i),...
        U_vector(1:nu,i), Pxx_filt(1:nx,1:nx,i), z_vector(1:ny,i+1),...
        Q_vector(1:nx,1:nx,i), R_vector(1:ny,1:ny,i+1));
end

end
