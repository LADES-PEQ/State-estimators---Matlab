function [Pxx_filt, x_filt, Cubature_points, Cubature_priori] = CCKFS(F, H,...
    X_vector, U_vector, Pxx_vector, z_vector, Q_vector, R_vector,...
    Constraints, Cubature_points, Cubature_priori, varargin)
% Calculation of the constrained cubature Kalman filter and smoother (CCKF&S)
% based on the forward-backward filtering approach proposed by Salau et al.
% (2012), in which the filtering and smoothing estimations follow the algorithms
% proposed by Zarei and Shokri (2014) and Arasaratnam and Haykin (2011), respectively.
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
% Constraints - Structure comprising the set of model constraints
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
% Arasaratnam, I., & Haykin, S. (2011). Cubature kalman smoothers. Automatica,
% 47(10), 2245-2250.
%
% Zarei, J., & Shokri, E. (2014). Nonlinear and constrained state estimation
% based on the cubature Kalman filter. Industrial & Engineering Chemistry
% Research, 53(10), 3938-3949
%
% Salau, N. P. G., Trierweiler, J. O. & Secchi, A. R. State estimators for
% better bioprocesses operation. Comput. Aided Chem. Eng. 30, 1267–1271,
% doi: 10.1016/B978-0-444-59520-1.50112-3 (2012).
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The numbering of equations refers to Arasaratnam and Haykin (2011).
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
        x_priori = x_priori +Wm*Cubature_priori(1:nx,j,N-i);
    end

    % Calculation of covariance matrices required to calculate the smoothing gain
    Pxx_cross = zeros(nx,nx);
    Pxx_priori = Q_vector(1:nx,1:nx,N-i);
    for j = 1:2*nx
        Pxx_cross = Pxx_cross+Wc*(Cubature_points(1:nx,j,N-i)-X_vector(1:nx,N-i))*...
            (Cubature_priori(1:nx,j,N-i)-x_priori)'; % Eq. 10
        Pxx_priori = Pxx_priori+Wc*(Cubature_priori(1:nx,j,N-i)-x_priori)*...
            (Cubature_priori(1:nx,j,N-i)-x_priori)'; % Eq. 11
        % OBS: Verificar equivalencia com as Eqs. 10 e 11
    end

    % Calculation of the smoothing gain
    inv_P = mldivide(Pxx_priori,eye(nx));
    inv_P = (inv_P+inv_P')/2;
    Ck = Pxx_cross*inv_P; % Eq. 21

    % Estimation of the smoothed counterparts of the states (x_smo), and 
    % the state error covariance matrix (Pxx_smo)
    x_smo(1:nx,N-i) = X_vector(1:nx,N-i) + Ck*(x_smo(1:nx,N-i+1)-x_priori); % Step 3 of Table 2  
    Pxx_smo(1:nx,1:nx,N-i) = Pxx_vector(1:nx,1:nx,N-i) +...
        Ck*(Pxx_smo(1:nx,1:nx,N-i+1)-Pxx_priori)*Ck'; % Eq. 22

    x_smo(1:nx,N-i) = Projection(x_smo(1:nx,N-i), U_vector(1:nu,N-i),...
        Constraints, Pxx_smo(1:nx,1:nx,N-i)); % Eq. 36
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
        Cubature_priori(1:nx,1:2*nx,i)] = CCKF(F,H,x_filt(1:nx,i),...
        U_vector(1:nu,i), Pxx_filt(1:nx,1:nx,i), z_vector(1:ny,i+1),...
        Q_vector(1:nx,1:nx,i), R_vector(1:ny,1:ny,i+1), Constraints);
end

end