function [Pxx_poster, x_poster, x_priori, phi] = CEKF_Simon(F, H, x_prior,...
    u, Pxx_prior, z, Q, R, Constraints, Jacob_x, Jacob_y, flag_stat, varargin)
% Calculation of the constrained extended Kalman filter (CEKF) based on the 
% methodology proposed by Simon and Chia (2002)
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
% Constraints - Structure comprising the set of model constraints
% Jacob_y - Numerical calculation of the state transition matrix
% Jacob_x - Measurement matrix
% 
% Outputs:
% Pxx_poster - State error covariance matrix at time k
% x_poster - State estimation at time k
% x_priori - State estimation based on information a priori at time k
% phi - State transition matrix at time k
%
% References:
% D. Simon, T.L. Chia, Kalman filtering with state equality constraints,
% IEEE Transactions on Aerospace and Electronic Systems 38 (2002) 128–136.
%
% Simon, D. (2006). Optimal state estimation: Kalman, H infinity, and nonlinear
% approaches. John Wiley & Sons.
%
% Ungarala, S., Dolence, E., & Li, K. (2007). Constrained extended Kalman 
% filter for nonlinear state estimation. IFAC Proceedings Volumes, 40(5), 63-68.
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The algorithm is equivalent to the extended Kalman filter,
% transcribed by Jazwinski (1970), if there are no active constraints in
% the projection function
%
% Note2: Simon and Chia (2002) defined the CEKF for a time-invariant system;
% thus, following the formulation of an asymptotically stable or stationary
% system in the Kalman filter. The parameter flag_stat eases the switch between 
% time-invariant (flag_stat = 1) and time-varying (flag_stat = 0) versions.
% 
% Note3: Simon (2006) detailed the assumptions and formulations of the steady
% state and time-varying Kalman filter in Sections 7.3 and 7.1, respectively.
%
% Note4: The variables x_priori and phi are returned to speed up computational
% time of a forward-backward filtering

% Definition of the number of states and measurements
nx = numel(x_prior);
ny = numel(z);

% Propagation of states based on the dynamic model
x_priori = F(x_prior(1:nx),u); % Eq. 85

% Calculation of the state transition matrix
phi = Jacob_x(x_prior(1:nx), u);

% Calculation of the a priori state error covariance matrix
if flag_stat == 1
    Pxx_priori = Pxx_prior; % Main assumption of the time-invariant formulation
else
    Pxx_priori = (phi*Pxx_prior*phi'+Q);
end

% Calculation of the Kalman gain
inv_K = mldivide(Jacob_y*Pxx_priori*Jacob_y'+R,eye(ny));
inv_K = (inv_K+inv_K')/2;
K = Pxx_priori*Jacob_y'*inv_K; % Eq. 86

% Update of state error covariance matrix based on information a posteriori
if flag_stat == 1
    Pxx_poster = phi*Pxx_prior*phi'-phi*K*Jacob_y*Pxx_prior*phi'+Q; % Eq. 88
else
    Pxx_poster = (eye(nx)-K*Jacob_y)*Pxx_priori*(eye(nx)-K*Jacob_y)'+K*R*K'; 
end
Pxx_poster = (Pxx_poster+Pxx_poster')/2;

% Update of state estimation based on information a posteriori
x_poster = x_priori + K*(z-H(x_priori)); % Eq. 87
x_poster = Projection(x_poster, u, Constraints, Pxx_poster);

end