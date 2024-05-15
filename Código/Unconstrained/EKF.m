function [Pxx_poster, x_poster, x_priori, phi] = EKF(F, H, x_prior, u, Pxx_prior,...
    z, Q, R, Jacob_x, Jacob_y, varargin)
% Calculation of the extended Kalman filter (EKF) based on the methodology 
% transcribed by Sarkka (2013)
%
% Inputs:
% F - State transition function
% H - Measurement function
% x_prior - State estimation based on information a posteriori at time k-1
% u - Set of time-invariant variables between sampling times k-1 and k
% Pxx_prior - State error covariance matrix at time k-1
% z - Measurements from the system at time k
% Q - Covariance matrix of the process noises
% R - Covariance matrix of the observation noises
% Jacob_x - Numerical calculation of the state transition matrix
% Jacob_y - Measurement matrix
% 
% Outputs:
% Pxx_poster - State error covariance matrix at time k
% x_poster - State estimation at time k
% x_priori - State estimation based on information a priori at time k
% phi - State transition matrix at time k
%
% References:
% Sarkka, S. (2013). Bayesian filtering and smoothing. Cambridge University Press
%
% Simon, D. (2006). Optimal state estimation: Kalman, H infinity, and nonlinear
% approaches. John Wiley & Sons.
% 
% Smith, G. L., Schmidt, S. F., & McGee, L. A. (1962) Application of statistical
% filter theory to the optimal estimation of position and velocity on board a
% circumlunar vehicle (Report 135). NASA
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The EKF was initially developed in unpublished works. The Smith et al.
% (1962) citation refers to one of the oldest technical documents from the 
% development group made public with the EKF formulation.
%
% Note2: The EKF might be formulated for time-varying (flag_stat = 0) or 
% steady-state (flag_stat = 1) filtering. Variable "flag_stat" is given to 
% switch between both approaches. Simon (2006) detailed their different 
% assumptions in Sections 7.3 and 7.1.
% 
% Note3: Equation numbers refer to Sarkka (2013).

flag_stat = 0;

% Definition of the number of states and measurements
nx = numel(x_prior);
ny = numel(z);

% Propagation of the state based on the dynamic model
x_priori = F(x_prior(1:nx),u); % Eq. 5.26 part 1

% Calculation of the state transition matrix
phi = Jacob_x(x_prior(1:nx), u);

% Calculation of the a priori state error covariance matrix
if flag_stat == 1
    Pxx_priori = Pxx_prior; % Main assumption of the time-invariant formulation
else
    Pxx_priori = (phi*Pxx_prior*phi'+Q); % Eq. 5.26 part 1
end

% Calculation of the Kalman gain
inv_K = mldivide(Jacob_y*Pxx_priori*Jacob_y'+R,eye(ny));
inv_K = (inv_K+inv_K')/2;
K = Pxx_priori*Jacob_y'*inv_K; % Eq. 5.27 part 1

% Update of state error covariance matrix based on information a posteriori
if flag_stat == 1
    Pxx_poster = phi*Pxx_prior*phi'-phi*K*Jacob_y'*Pxx_prior*phi'+Q; 
else
    Pxx_poster = (eye(nx)-K*Jacob_y)*Pxx_priori*(eye(nx)-K*Jacob_y)'+K*R*K'; % Eq. 5.27 part 2
end
Pxx_poster = (Pxx_poster+Pxx_poster')/2;

% Update of state estimation based on information a posteriori
x_poster = x_priori + K*(z-H(x_priori)); % Eq. 5.27 part 3

end