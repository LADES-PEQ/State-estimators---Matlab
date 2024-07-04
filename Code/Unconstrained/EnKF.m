function [Pxx_poster, x_poster, Ensemble_poster, Ensemble_priori] = EnKF(F,...
    H, Ensemble_prior, u, z, Q, R, Jacob_y, varargin)
% Calculation of the ensemble Kalman filter (EnKF) based on the methodology
% proposed by Eversen (1994), transcribed with further detailing by Eversen (2003)
%
% Inputs:
% F - State transition function
% H - Measurement function
% Ensemble_prior - State estimations at time k-1 for each ensemble member
% u - Set of time-invariant variables between sampling times k-1 and k
% z - Measurements from the system at time k
% Q - Covariance matrix of the process noises
% R - Covariance matrix of the observation noises
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
% Evensen, G. (2003). The ensemble Kalman filter: Theoretical formulation 
% and practical implementation. Ocean dynamics, 53, 343-367.
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The numbering of equations refers to Eversen (2003).
% 
% Note2: The variables with the prefix "Ensemble" have calculations for each 
% ensemble member propagated based on the column position.
%
% Note3: The ensemble Kalman filter does not require prior knowledge about 
% measurement noises; however, R is usually defined for the observer. Current
% implementation estimates R in each sampling time for an empty input.
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
vk = zeros(ny,n_ensemble);
Ensemble_poster = zeros(nx,n_ensemble);

Q_chol = chol(Q,"lower");
for i = 1:n_ensemble
    wk = Q_chol*randn(nx,1);
    % Propagation of ensemble members based on the dynamic model
    Ensemble_priori(1:nx,i) = F(Ensemble_prior(1:nx,i),u)+wk; % Eq. 88
    
    % Calculation of the ensemble mean
    x_priori = x_priori + Wm*Ensemble_priori(1:nx,i); % Eq. 45

    % Calculation of measurements for each ensemble member
    Ensemble_output(1:ny,i) = H(Ensemble_prior(1:nx,i)); 
    if isempty(R)
        vk(1:ny,i) = z - Ensemble_output(1:ny,i)+H(wk); % Eq. 90 rearranged
    end
end

% Calculation of the ensemble perturbation matrix (A' in Eversen (2003))
Pxx_priori = zeros(nx,nx);
for i = 1:n_ensemble
   Pxx_priori = Pxx_priori+Wc*(Ensemble_priori(1:nx,i)-x_priori)*...
       (Ensemble_priori(1:nx,i)-x_priori)'; % Eq. 47
   % OBS: (Ensemble_priori(1:nx,i)-x_priori) is equivalent to Eq. 46
end

% Calculation of a sample covariance matrix of observation noises if R was 
% not provided
if isempty(R)
   R = Wc*vk*vk'; % Eq. 51
end

% Calculation of the Kalman gain
inv_K = mldivide(Jacob_y*Pxx_priori*Jacob_y'+R,eye(ny));
inv_K = (inv_K+inv_K')/2;
K = Pxx_priori*Jacob_y'*inv_K; % Eq. 23

% Update of estimation from each ensemble member based on information a posteriori;
R_chol = chol(R,"lower");
for i = 1:n_ensemble
    Ensemble_poster(1:nx,i) = Ensemble_priori(1:nx,i)+...
        K*(z+R_chol*randn(ny,1)-Ensemble_output(1:ny,i)); % Eq. 52    
end

% Update of the state estimation based on information a posteriori
x_poster = x_priori + K*(z-H(x_priori)); % Eq. 21
% x_poster = zeros(nx,1);
% for i = 1:n_ensemble
%     x_poster = x_poster + Wm*Ensemble_poster(1:nx,i);
% end

% OBS: Eversen (2003) mentioned that the calculation of x_poster refers to 
% an arbitrary choice between commented and uncommented formulations presented
% above. 

% Update of state error covariance matrix based on information a posteriori
Pxx_poster = (eye(nx)-K*Jacob_y)*Pxx_priori*(eye(nx)-K*Jacob_y)'+K*R*K'; % Eq. 24
Pxx_poster = (Pxx_poster+Pxx_poster')/2;

end