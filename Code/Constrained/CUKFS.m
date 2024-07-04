function [Pxx_filt, x_filt, Sigma_points, Sigma_priori] = CUKFS(F, H,...
    X_vector, U_vector, Pxx_vector, z_vector, Q_vector, R_vector,...
    Constraints, Sigma_points, Sigma_priori, varargin)
% Calculation of the constrained unscented Kalman filter and smoother (CUKF&S) 
% based on the forward-backward filtering approach proposed by Salau et al.
% (2012), in which the filtering and smoothing estimations follow the algorithms 
% proposed by Kandepu et al. (2008) and Sarkka (2008), respectively.
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
% Sigma_points - Set of sigma points defined between k-N-1 and k-1
% Sigma_priori - A priori estimations for each sigma point between k-N and k-1
%
% Outputs:
% Pxx_filt - State error covariance matrices between k-N-1 and k
% x_filt - State estimations between k-N-1 and k
% Sigma_points - Set of sigma points defined between k-N-1 and k-1
% Sigma_priori - A priori estimations for each sigma point between k-N and k
%
% References:
% Kandepu, R., Foss, B., & Imsland, L. (2008). Applying the unscented Kalman
% filter for nonlinear state estimation. Journal of process control, 18(7-8),
% 753-768.
%
% S. Sarkka, Unscented Rauch--Tung--Striebel Smoother, in IEEE Transactions
% on Automatic Control, vol. 53, no. 3, pp. 845-849, April 2008,
% doi: 10.1109/TAC.2008.919531
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The numbering of equations refers to Sarkka (2008).
% 
% Note2: The variables with the prefix "Sigma" have calculations for each 
% sigma point propagated based on the column position.

% Definition of the smoothing horizon and the number of states,
% measurements, inputs and augmented states
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

% Definition of tuning parameters of the unscented tranformation
Alfa = 1; % Scaling parameter ranging in [0,1]
Kappa = 0; % Secondary scaling parameter
Beta = 2; % Distribution parameter, whose value Beta = 2 implies a normal distribution

% Calculation of intermediate parameters
Lambda = Alfa^2*(nx+Kappa)-nx;
Gamma = sqrt(nx+Lambda);
Wm = [Lambda/(nx+Lambda) repmat(1/(2*(nx+Lambda)),1,2*nx)];
Wc = Wm;
Wc(1) = Wc(1)+(1-Alfa^2+Beta);

% Smoothing estimations from k-2 to k-N-1
for i = 1:N-1   
    % Calculation of the state estimations a priori 
    x_priori = zeros(nx,1);
    for j = 1:2*nx+1
        x_priori = x_priori+Wm(j)*Sigma_priori(1:nx,j,N-i);
    end
%     x_priori = Projection(x_priori, Constraints,...
%         Pxx_vector(1:nx,1:nx,N-i)); % Eq. 36

    % Calculation of covariance matrices required to calculate the smoothing gain
    Pxx_priori = Q_vector(1:nx,1:nx,N-i);
    Pxx_cross = zeros(nx,nx);
    for j = 1:2*nx+1
        Pxx_priori = Pxx_priori+Wc(j)*(Sigma_priori(1:nx,j,N-i)-x_priori)*...
            (Sigma_priori(1:nx,j,N-i)-x_priori)';
        Pxx_cross = Pxx_cross+Wc(j)*(Sigma_points(1:nx,j,N-i)-X_vector(1:nx,N-i))*...
            (Sigma_priori(1:nx,j,N-i)-x_priori)';
    end

    % Calculation of the smoothing gain
    inv_Ck = mldivide(Pxx_priori,eye(nx));
    inv_Ck = (inv_Ck+inv_Ck')/2;
    Ck = Pxx_cross*inv_Ck;

    % Estimation of the smoothed counterparts of the states (x_smo), and 
    % the state error covariance matrix (Pxx_smo)
    x_smo(1:nx,N-i) = X_vector(1:nx,N-i) + Ck*(x_smo(1:nx,N-i+1)-x_priori);
    Pxx_smo(1:nx,1:nx,N-i) = Pxx_vector(1:nx,1:nx,N-i) +...
        Ck*(Pxx_smo(1:nx,1:nx,N-i+1)-Pxx_priori)*Ck';

    x_smo(1:nx,N-i) = Projection(x_smo(1:nx,N-i), U_vector(1:nu,N-i),...
        Constraints, Pxx_smo(1:nx,1:nx,N-i)); % Eq. 36
end

% Memory allocation for output variables of the forward filtering pass
x_filt = zeros(nx,N+1);
Pxx_filt = zeros(nx,nx,N+1);
Sigma_points = zeros(nx,2*nx+1,N+1);
Sigma_priori = zeros(nx,2*nx+1,N+1);

% Definition of the initial condition of the forward filtering pass by the
% smoothed estimation at k-N-1
x_filt(1:nx,1) = x_smo(1:nx,1);
Pxx_filt(1:nx,1:nx,1) = Pxx_smo(1:nx,1:nx,1);

% Filtering estimations from k-N to k
for i = 1:N
    [Pxx_filt(1:nx,1:nx,i+1),x_filt(1:nx,i+1), Sigma_points(1:nx,1:2*nx+1,i),...
        Sigma_priori(1:nx,1:2*nx+1,i)] = CUKF(F, H, x_filt(1:nx,i),...
        U_vector(1:nu,i), Pxx_filt(1:nx,1:nx,i), z_vector(1:ny,i+1),...
        Q_vector(1:nx,1:nx,i), R_vector(1:ny,1:ny,i+1), Constraints);
end
Sigma_points(nx+1:end,:,:) = [];
end