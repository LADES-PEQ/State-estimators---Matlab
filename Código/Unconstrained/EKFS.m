function [Pxx_filt, x_filt, x_priori, phi] = EKFS(F, H , X_vector, U_vector,...
    Pxx_vector, z_vector, Q_vector, R_vector, Jacob_x, Jacob_y,...
    x_priori, phi, varargin)
% Calculation of the extended Kalman filter and smoother (EKF&S) for a 
% forward-backward filtering approach, in which the filtering and smoothing
% estimations follow the algorithms transcribed by Sakka (2013)
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
% Jacob_x - Numerical calculation of the state transition matrix
% Jacob_y - Measurement matrix
% x_priori - A priori state estimations between k-N and k-1
% phi - State transition matrices between k-N and k-1
% 
% Outputs:
% Pxx_filt - State error covariance matrices between k-N-1 and k
% x_filt - State estimations between k-N-1 and k
% x_priori - A priori state estimations between k-N and k
% phi - State transition matrices between k-N and k
%
% References:
% Rauch, H. E., Tung, F. & Striebel, C. T. Maximum likelihood estimates of 
% linear dynamic systems. AIAA J. 3, 1445–1450, doi: 10.2514/3.3166 (1965).
% 
% Sarkka, S. (2013). Bayesian filtering and smoothing. Cambridge University Press
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note: The smoothing algorithm might be found in the literature as the extended
% Rauch-Tung-Striebel smoother (RTS) or the extended Kalman smoother. The 
% first nomenclature refer to its paper from Rauch et al. (1965), while
% the latter refer to the filtering counterpart.
% 
% numbering of equations refers to Rauch et al. (1965).

% Definition of the smoothing horizon and the number of states and measurements
[nx,N] = size(X_vector);
ny = size(z_vector,1);
nu = size(U_vector,1);

% Memory allocation for output variables of the smoothing estimations
x_smo = zeros(nx,N);
Pxx_smo = zeros(nx,nx,N);

% Definition of initial conditions for the smoothing estimations based on 
% estimations a posteriori at time k-1
x_smo(1:nx,N) = X_vector(1:nx,N);
Pxx_smo(1:nx,1:nx,N) = Pxx_vector(1:nx,1:nx,N);

% Smoothing estimations from k-2 to k-N-1
for i = 1:N-1
    % Calculation of covariance matrices required to calculate the smoothing gain
    Pxx_priori = phi(1:nx,1:nx,N-i)*Pxx_vector(1:nx,1:nx,N-i)*phi(1:nx,1:nx,N-i)'+...
        Q_vector(1:nx,1:nx,N-i); % Eq. 9.2 part 1
    
    % Calculation of the smoothing gain
    inv_pri = mldivide(Pxx_priori,eye(nx));
    inv_pri = (inv_pri+inv_pri')/2;
    Ck = Pxx_vector(1:nx,1:nx,N-i)*phi(1:nx,1:nx,N-i)'*inv_pri; % Eq. 9.2 part 2
    
    % Estimation of the smoothed counterparts of the states (x_smo), and 
    % the state error covariance matrix (Pxx_smo)
    x_smo(1:nx,N-i) = X_vector(1:nx,N-i) +...
        Ck*(x_smo(1:nx,N-i+1)-x_priori(1:nx,N-i)); % Eq. 9.2 part 3
    Pxx_smo(1:nx,1:nx,N-i) = Pxx_vector(1:nx,1:nx,N-i) +...
        Ck*(Pxx_smo(1:nx,1:nx,N-i+1)-Pxx_priori)*Ck'; % Eq. 9.2 part 4
end

% Memory allocation for output variables of the forward filtering pass
x_filt = zeros(nx,N+1);
Pxx_filt = zeros(nx,nx,N+1);
x_priori = zeros(nx,N+1);
phi = zeros(nx,nx,N+1);

% Definition of the initial condition of the forward filtering pass by the
% smoothed estimation at k-N-1
x_filt(1:nx,1) = x_smo(1:nx,1);
Pxx_filt(1:nx,1:nx,1) = Pxx_smo(1:nx,1:nx,1);

% Filtering estimations from k-N to k
for i = 1:N
    [Pxx_filt(1:nx,1:nx,i+1), x_filt(1:nx,i+1), x_priori(1:nx,i),...
        phi(1:nx,1:nx,i)] = EKF(F, H, x_filt(1:nx,i), U_vector(1:nu,i),...
        Pxx_filt(1:nx,1:nx,i), z_vector(1:ny,i+1), Q_vector(1:nx,1:nx,i),...
        R_vector(1:ny,1:ny,i+1), Jacob_x, Jacob_y);
end

end
