function [Pxx_poster, x_poster, Ensemble_poster, Ensemble_priori] = CEnKF(F,...
    H, Ensemble_prior, u, z, Q, R, Constraints, Jacob_y, Pxx_prior, varargin)
% Calculation of the constrained ensemble Kalman filter (CEnKF) based on the
% methodology proposed by Pan and Wood (2006)
%
% Inputs:
% F - State transition function
% H - Measurement function
% Ensemble_prior - State estimations at time k-1 for each ensemble member
% u - Set of time-invariant variables between sampling times k-1 and k
% z - Measurements from the system at time k
% Q - Covariance matrix of the process noises
% R - Covariance matrix of the observation noises
% Constraints - Structure comprising the set of model constraints
% Jacob_y - Jacobian of the state transition function
% 
% Outputs:
% Pxx_poster - State error covariance matrix at time k
% x_poster - State estimation at time k
% Ensemble_poster - State estimations at time k for each ensemble member
% Ensemble_priori - A priori state estimation for each ensemble member at time k
%
% References:
% Evensen, G., Sequential data assimilation with a nonlinear quasi-geostrophic
% model using Monte Carlo methods to forecast error statistics, J. Geophys.
% Res., 99, 10,143–10,162, 1994.
%
% Pan, M., & Wood, E. F. (2006). Data assimilation for estimating the 
% terrestrial water budget using a constrained ensemble Kalman filter. 
% Journal of Hydrometeorology, 7(3), 534-547. 
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The algorithm is equivalent to the ensemble Kalman filter,
% proposed by Evensen (1994), if there are no active constraints in
% the projection function
% 
% Note2: The variables with the prefix "Ensemble" have calculations for each 
% ensemble member propagated based on the column position.
%
% Note3: The variable Ensemble_priori is returned to speed up computational
% time of a forward-backward filtering.

% Definition of the number of states, measurements and ensemble members
[nx,n_ensemble] = size(Ensemble_prior);
ny = numel(z);

% Definition of weight matrices
Wm = 1/n_ensemble;
Wc = 1/(n_ensemble-1);

% Memory allocation for estimations based information a priori
Ensemble_priori = zeros(nx,n_ensemble);
x_priori = zeros(nx,1);
Ensemble_output = zeros(ny,n_ensemble);
Ensemble_poster = zeros(nx,n_ensemble);

Q_chol = chol(Q,"lower");
for i = 1:n_ensemble
    wk = Q_chol*randn(nx,1);
    % Propagation of ensemble members based on the dynamic model
    Ensemble_priori(1:nx,i) = F(Ensemble_prior(1:nx,i),u)+wk; % Eq. 8
    
    % Calculation of the ensemble mean
    x_priori = x_priori + Wm*Ensemble_priori(1:nx,i); 

    % Calculation of measurements for each ensemble member
    Ensemble_output(1:ny,i) = H(Ensemble_prior(1:nx,i)); 
end

% Calculation of the ensemble perturbation matrix
Pxx_priori = zeros(nx,nx);
for i = 1:n_ensemble
   Pxx_priori = Pxx_priori+Wc*(Ensemble_priori(1:nx,i)-x_priori)*...
       (Ensemble_priori(1:nx,i)-x_priori)'; 
end

% Calculation of the Kalman gain
inv_K = mldivide(Jacob_y*Pxx_priori*Jacob_y'+R,eye(ny));
inv_K = (inv_K+inv_K')/2;
K = Pxx_priori*Jacob_y'*inv_K; % Eq. 6

% Update of state error covariance matrix based on information a posteriori
Pxx_poster = (eye(nx)-K*Jacob_y)*Pxx_priori*(eye(nx)-K*Jacob_y)'+K*R*K'; 
Pxx_poster = (Pxx_poster+Pxx_poster')/2;

% Update of estimation from each ensemble member based on information a posteriori;
R_chol = chol(R,"lower");
for i = 1:n_ensemble
    Ensemble_poster(1:nx,i) = Ensemble_priori(1:nx,i)+...
        K*(z+R_chol*randn(ny,1)-Ensemble_output(1:ny,i)); % Eq. 5
    Ensemble_poster(1:nx,i) = Projection(Ensemble_poster(1:nx,i), u,...
        Constraints, Pxx_poster); % Eq. 17
end
% Update of the state estimation based on information a posteriori
x_poster = x_priori + K*(z-H(x_priori)); % Eq. 5
x_poster = Projection(x_poster, u, Constraints, Pxx_poster); % Eq. 15

end
