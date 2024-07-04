function [Pxx_filt, x_filt, Ensemble_filt, Ensemble_priori] = CEnKFS(F, H,...
    Ensemble_vector, U_vector, Pxx_vector, z_vector, Q_vector, R_vector,...
    Constraints, Jacob_y, X_vector, Ensemble_priori, varargin)
% Calculation of the ensemble Kalman filter and smoother (EnKF&S) based on 
% the forward-backward filtering approach proposed by Salau et al. (2012), 
% in which the filtering and smoothing estimations follow the algorithms 
% proposed by Evensen (2003) and Raanes (2016), respectively.
%
% Inputs:
% F - State transition function
% H - Measurement function
% Ensemble_vector - State estimations between k-N-1 and k-1 for each ensemble member
% U_vector - Set of time-invariant variables between k-N-1 and k-1
% Pxx_vector - State error covariance matrices between k-N-1 and k-1
% z_vector - Measurements from the system between k-N-1 and k
% Q_vector - Covariance matrices of the process noises between k-N-1 and k-1
% R_vector - Covariance matrices of the observation noises between k-N-1 and k
% Constraints - Structure comprising the set of model constraints
% Jacob_y - Measurement matrix
% X_vector - State estimations between k-N-1 and k-1
% Ensemble_priori - A priori estimations for each ensemble member between k-N and k-1
%
% Outputs:
% Pxx_filt - State error covariance matrices between k-N-1 and k
% x_filt - State estimations between k-N-1 and k
% Ensemble_filt - State estimations between k-N-1 and k for each ensemble member
% Ensemble_priori - A priori estimations for each ensemble member between k-N and k
%
% References:
% Evensen, G. (2003). The ensemble Kalman filter: Theoretical formulation
% and practical implementation. Ocean dynamics, 53, 343-367.
%
% Raanes, P.N. (2016), On the ensemble Rauch-Tung-Striebel smoother and its
% equivalence to the ensemble Kalman smoother. Q.J.R. Meteorol. Soc., 142:
% 1259-1264. https://doi.org/10.1002/qj.2728
%
% Salau, N. P. G., Trierweiler, J. O. & Secchi, A. R. State estimators for 
% better bioprocesses operation. Comput. Aided Chem. Eng. 30, 1267–1271, 
% doi: 10.1016/B978-0-444-59520-1.50112-3 (2012).
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The numbering of equations refers to Raanes (2016).
% 
% Note2: The variables with the prefix "Ensemble" have calculations for each 
% ensemble member propagated based on the column position.
%
% Note3: The execution time of the forward-backward filtering is extremely 
% slow for a large number of particles. In addition, preliminary tests found
% that model accuracy is more sensitive to particle size than the smoothing
% horizon. Hence, this algorithm is mainly theoretical or intended for severe
% fine-tuning of the observer.

% Definition of the smoothing horizon and the number of states, measurements
% and ensemble members
[nx, n_ensemble, N] = size(Ensemble_vector);
ny = size(z_vector,1);
nu = size(U_vector,1);

% Calculation of intermediate parameters
Wm = 1/n_ensemble;
Wc = 1/(n_ensemble-1);

% Memory allocation for output variables of the smoothing estimations
Ensemble_smo = zeros(nx,n_ensemble,N);
x_smo = zeros(nx,N);
Pxx_smo = zeros(nx,nx,N);

% Definition of initial conditions for the smoothing estimations based on 
% estimations a posteriori at time k-1
Ensemble_smo(1:nx,1:n_ensemble,N) = Ensemble_vector(1:nx,1:n_ensemble,N);
x_smo(1:nx,N) = X_vector(1:nx,N);
Pxx_smo(1:nx,1:nx,N) = Pxx_vector(1:nx,1:nx,N);

% Smoothing estimations from k-2 to k-N-1
for i = 1:N-1
    % Calculation of the state estimations a priori 
    x_priori = zeros(nx,1);
    for j = 1:n_ensemble
        x_priori = x_priori + Wm*Ensemble_priori(1:nx,j,N-i);
    end

    % Calculation of covariance matrices required to calculate the smoothing gain
    Pxx_cross = zeros(nx,nx);
    Pxx_priori = zeros(nx,nx);
    for j = 1:n_ensemble
        Pxx_cross = Pxx_cross+Wc*(Ensemble_vector(1:nx,j,N-i)-X_vector(1:nx,N-i))*...
            (Ensemble_priori(1:nx,j,N-i)-x_priori)';
        % OBS: (Ensemble_vector(1:nx,j,N-i)-X_vector(1:nx,N-i)) is equivalent to
        % A_{t|t} from Raanes (2016)
        Pxx_priori = Pxx_priori+Wc*(Ensemble_priori(1:nx,j,N-i)-x_priori)*...
            (Ensemble_priori(1:nx,j,N-i)-x_priori)';
        % OBS: (Ensemble_priori(1:nx,j,N-i)-x_priori) is equivalent to
        % A_{t+1|t} from Raanes (2016)
    end

    % Calculation of the smoothing gain
    inv_Ck = mldivide(Pxx_priori,eye(nx));
    % OBS: Raanes (2016) followed a pseudoinverse of Pxx_priori
    inv_Ck = (inv_Ck+inv_Ck')/2;
    Ck = Pxx_cross*inv_Ck; % Eq. 16

    % Estimation of the smoothed counterparts of the states (x_smo), and 
    % the state error covariance matrix (Pxx_smo)
    Pxx_smo(1:nx,1:nx,N-i) = Pxx_vector(1:nx,1:nx,N-i) +...
        Ck*(Pxx_smo(1:nx,1:nx,N-i+1)-Pxx_priori)*Ck'; % Eq. 11
    for j = 1:n_ensemble
        Ensemble_smo(1:nx,j,N-i) = Ensemble_vector(1:nx,j,N-i)+...
            Ck*(Ensemble_smo(1:nx,j,N-i+1)-Ensemble_priori(1:nx,j,N-i)); % Eq. 15
        Ensemble_smo(1:nx,j,N-i) = Projection(Ensemble_smo(1:nx,j,N-i),...
            U_vector(1:nu,N-i), Constraints, Pxx_smo(1:nx,1:nx,N-i)); % Eq. 15
        
%         x_smo(1:nx,N-i) = x_smo(1:nx,N-i) + Wm*Ensemble_smo(1:nx,j,N-i);
%         x_smo(1:nx,N-i) = Projection(x_smo(1:nx,N-i), Constraints,...
%             Pxx_smo(1:nx,1:nx,N-i)); % Eq. 15
    end
   x_smo(1:nx,N-i) = X_vector(1:nx,N-i) + Ck*(x_smo(1:nx,N-i+1)-x_priori); % Step 3 of Table 2  
    x_smo(1:nx,N-i) = Projection(x_smo(1:nx,N-i), U_vector(1:nu,N-i),...
        Constraints, Pxx_smo(1:nx,1:nx,N-i)); % Eq. 36
end

% Memory allocation for output variables of the forward filtering pass
x_filt = zeros(nx,N+1);
Ensemble_filt = zeros(nx,n_ensemble,N+1);
Ensemble_priori = zeros(nx,n_ensemble,N+1);
Pxx_filt = zeros(nx,nx,N+1);

% Definition of the initial condition of the forward filtering pass by the
% smoothed estimation at k-N-1
x_filt(1:nx,1) = x_smo(1:nx,1);
Ensemble_filt(1:nx,1:n_ensemble,1) = Ensemble_smo(1:nx,1:n_ensemble,1);
Pxx_filt(1:nx,1:nx,1) = Pxx_smo(1:nx,1:nx,1);

% Filtering estimations from k-N to k
for i = 1:N
    [Pxx_filt(1:nx,1:nx,i+1), x_filt(1:nx,i+1), Ensemble_filt(1:nx,1:n_ensemble,i+1),...
        Ensemble_priori(1:nx,1:n_ensemble,i)] = CEnKF(F, H, Ensemble_filt(1:nx,1:n_ensemble,i),...
        U_vector(1:nu,i), z_vector(1:ny,i+1), Q_vector(1:nx,1:nx,i),...
        R_vector(1:ny,1:ny,i+1), Constraints, Jacob_y, Pxx_filt(1:nx,1:nx,i));
end

end