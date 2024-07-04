function [Pxx_poster, x_poster, Sigma_points, Sigma_priori] = UKF(F, H,...
    x_prior, u, Pxx_prior, z, Q, R, varargin)
% Calculation of the square-root unscented Kalman filter (UKF) based on the
% methodology proposed by Wan and van der Merwe (2001) and transcribed by 
% Sarkka (2013)
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
% 
% Outputs:
% Pxx_poster - State error covariance matrix at time k
% x_poster - State estimation at time k
% Sigma_points - Set of sigma points defined around state estimation at time k
% Sigma_priori - A priori state estimation for each sigma point at time k
%
% References:
%
% Julier, S. J., & Uhlmann, J. K. (1997, July). New extension of the Kalman
% filter to nonlinear systems. In Signal processing, sensor fusion, and target
% recognition VI (Vol. 3068, pp. 182-193). Spie.
% 
% Van Der Merwe, R., & Wan, E. A. (2001, May). The square-root unscented 
% Kalman filter for state and parameter-estimation. In 2001 IEEE international
% conference on acoustics, speech, and signal processing. Proceedings (Cat.
% No. 01CH37221) (Vol. 6, pp. 3461-3464). IEEE.
% 
% Sarkka, S. (2013). Bayesian filtering and smoothing. Cambridge University Press
%
% E. A. Wan and R. Van der Merwe, The unscented Kalman filter for nonlinear
% estimation, Proceedings of the IEEE 2000 Adaptive Systems for Signal 
% Processing, Communications, and Control Symposium (Cat. No.00EX373),
% Lake Louise, AB, Canada, 2000, pp. 153-158, doi: 10.1109/ASSPCC.2000.882463.
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The variables with the prefix "Sigma" have calculations for each 
% sigma point propagated based on the column position.
%
% Note2: The UKF was initially proposed by Julier and Uhlmann (1997); however,
% early extensions became equally popular, so that the implemented algorithm 
% and the algorithm proposed by Wan and Van der Merwe (2000) can also be found 
% in the literature referred to as UKF.
%
% Note3: Equation numbers refer to Sarkka (2013).
%
% Note4: The variables Sigma_points and Sigma_priori are returned to speed
% up computational time of a forward-backward filtering.

% Definition of the number of states and measurements
nx = numel(x_prior);
ny = numel(z);

% Memory allocation for estimations based information a priori
Sigma_priori = zeros(nx,2*nx+1);
Sigma_output = zeros(ny,2*nx+1);
x_priori = zeros(nx,1);
y_priori = zeros(ny,1);

% Definition of tuning parameters of the unscented tranformation
Alfa = 0.001; % Scaling parameter ranging in [0,1]
Kappa = 0; % Secondary scaling parameter, which must be positive to guarantee a 
% positive definite state error covariance matrix
Beta = 2; % Distribution parameter, whose value Beta = 2 is optimal for a normal distribution

% Calculation of intermediate parameters
Lambda = Alfa^2*(nx+Kappa)-nx;
Gamma = sqrt(nx+Lambda);
Wm = [Lambda/(nx+Lambda) repmat(1/(2*(nx+Lambda)),1,2*nx)];
Wc = Wm;
Wc(1) = Wc(1)+(1-Alfa^2+Beta);

% Definition of a set of sigma points around x_prior
Sigma_points = [x_prior x_prior+Gamma*chol(Pxx_prior,'lower')...
                x_prior-Gamma*chol(Pxx_prior,'lower')]; % Eq. 5.84

% Propagation of sigma points based on the dynamic model
for i = 1:2*nx+1
    Sigma_priori(1:nx,i) = F(Sigma_points(1:nx,i),u); % Eq. 5.85
    x_priori = x_priori+Wm(i)*Sigma_priori(1:nx,i); % Eq. 5.86 part 1
end

% Calculation of the a priori state error covariance matrix
Pxx_priori = Q;
for i = 1:2*nx+1
    Pxx_priori = Pxx_priori+Wc(i)*(Sigma_priori(1:nx,i)-x_priori)*...
        (Sigma_priori(1:nx,i)-x_priori)'; % Eq. 5.86 part 2
end

% Definition of a set of cubature points around x_priori

Sigma_update = [x_priori x_priori+Gamma*chol(Pxx_priori,'lower')...
                x_priori-Gamma*chol(Pxx_priori,'lower')]; % Eq. 5.87
% Sigma_update = Sigma_priori;

% Calculation of measurements for each sigma point
for i = 1:2*nx+1
    Sigma_output(1:ny,i) = H(Sigma_update(1:nx,i)); % Eq. 5.88
    y_priori = y_priori+Wm(i)*Sigma_output(1:ny,i); % Eq. 5.89 part 1
end

% Calculation of the covariance matrix of measurement error (Pyy) and the 
% cross covariance matrix of the state and measurement errors (Pxy)
Pyy = R; % Eq. 5.89 part 2
Pxy = zeros(nx,ny); 
for i = 1:2*nx+1
    Pyy = Pyy + Wc(i)*(Sigma_output(1:ny,i)-y_priori)*...
        (Sigma_output(1:ny,i)-y_priori)'; % Eq. 5.89 part 3
    Pxy = Pxy + Wc(i)*(Sigma_update(1:nx,i)-x_priori)*...
        (Sigma_output(1:ny,i)-y_priori)'; % Eq. 5.89 part 4
end

% Calculation of the Kalman gain
inv_Pyy = mldivide(Pyy,eye(ny));
inv_Pyy = (inv_Pyy+inv_Pyy')/2;
K = Pxy*inv_Pyy; % Eq. 5.90 part 1

% Update of state estimation based on information a posteriori
innov = z-y_priori;
x_poster = x_priori+K*innov; % Eq. 5.90 part 2

% Update of state error covariance matrix based on information a posteriori
Pxx_poster = Pxx_priori-K*Pyy*K'; % Eq. 5.90 part 3
Pxx_poster = (Pxx_poster+Pxx_poster')/2;

end