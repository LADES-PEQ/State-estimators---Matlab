function [Pxx_poster, x_poster, Cubature_points, Cubature_priori] =...
    CCKF(F, H, x_prior, u, Pxx_prior, z, Q, R, Constraints, varargin)
% Calculation of the constrained cubature Kalman filter (CCKF) based on the
% methodology proposed by Zarei and Shokri (2014)
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
%
% Outputs:
% Pxx_poster - State error covariance matrix at time k
% x_poster - State estimation at time k
% Cubature_points - Set of cubature points defined around state estimation at time k
% Cubature_priori - A priori state estimation for each cubature points at time k
%
% References:
% Arasaratnam, I., Haykin, S. (2009). Cubature kalman filters. IEEE
% Transactions on automatic control, 54(6), 1254-1269.
%
% Zarei, J., & Shokri, E. (2014). Nonlinear and constrained state estimation
% based on the cubature Kalman filter. Industrial & Engineering Chemistry
% Research, 53(10), 3938-3949
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The algorithm is equivalent to the cubature Kalman filter, proposed
% by Arasaratnam and Haykin (2009), if there are no active constraints in the
% projection function.
%
% Note2: The variables with the prefix "Cubature" have calculations for each
% cubature point propagated based on the column position.
% 
% Note3: The variables Cubature_points and Cubature_priori are returned to 
% speed up computational time of a forward-backward filtering.

% Definition of the number of states and measurements
nx = numel(x_prior);
ny = numel(z);

% Memory allocation for estimations based information a priori
Cubature_points = zeros(nx,2*nx);
Cubature_priori = zeros(nx,2*nx);
Cubature_update = zeros(nx,2*nx);
Cubature_output = zeros(ny,2*nx);
x_priori = zeros(nx,1);
y_priori = zeros(ny,1);

% Definition of a set of cubature points around x_prior
Pxx_prior_chol = chol(Pxx_prior,'lower');
Zeta = sqrt(nx)*eye(nx);
for i = 1:nx
    Cubature_points(:,i) = x_prior + Pxx_prior_chol*Zeta(1:nx,i); % Eq. 17 part 1
    Cubature_points(:,i+nx) = x_prior - Pxx_prior_chol*Zeta(1:nx,i); % Eq. 17 part 2
    Cubature_points(:,i) = Projection(Cubature_points(:,i), u,...
        Constraints, Pxx_prior); % Eq. 36 part 1
    Cubature_points(:,i+nx) = Projection(Cubature_points(:,i+nx), u,...
        Constraints, Pxx_prior); % Eq. 36 part 2
end

% Definition of weight matrices
Wm = 1/(2*nx);
Wc = 1/(2*nx);

% Propagation of cubature points based on the dynamic model
for i = 1:2*nx
    Cubature_priori(1:nx,i) = F(Cubature_points(1:nx,i), u); % Eq. 17
    x_priori = x_priori+Wm*Cubature_priori(1:nx,i); % Eq. 18
end

% Calculation of the a priori state error covariance matrix
Pxx_priori = Q;
for i = 1:2*nx
    Pxx_priori = Pxx_priori+Wc*(Cubature_priori(1:nx,i)-x_priori)*...
        (Cubature_priori(1:nx,i)-x_priori)'; % Eq. 19
end


% Definition of a set of cubature points around x_priori
Pxx_priori_chol = chol(Pxx_priori,'lower');
for i = 1:nx
    Cubature_update(:,i) = x_priori + Pxx_priori_chol*Zeta(1:nx,i); % Eq. 20 part 1
    Cubature_update(:,i+nx) = x_priori - Pxx_priori_chol*Zeta(1:nx,i); % Eq. 20 part 2
end

% Calculation of measurements for each cubature point
for i = 1:2*nx
    Cubature_output(1:ny,i) = H(Cubature_update(1:nx,i)); % Eq. 21
    y_priori = y_priori+Wm*Cubature_output(1:ny,i); % Eq. 22
end

% Initialization of the covariance matrices
Pyy = R; % Eq. 23 part 1
Pxy = zeros(nx,ny);

% Calculation of the covariance matrix of measurement error (Pyy) and the
% cross covariance matrix of the state and measurement errors (Pxy)
for i = 1:2*nx
    Pyy = Pyy + Wc*(Cubature_output(1:ny,i)-y_priori)*...
        (Cubature_output(1:ny,i)-y_priori)'; % Eq. 23 part 2
    Pxy = Pxy + Wc*(Cubature_update(1:nx,i)-x_priori)*...
        (Cubature_output(1:ny,i)-y_priori)'; % Eq. 24
end

% Calculation of the Kalman gain
inv_Pyy = mldivide(Pyy,eye(ny));
inv_Pyy = (inv_Pyy+inv_Pyy')/2;
K = Pxy*inv_Pyy; % Eq. 25

% Update of state error covariance matrix based on information a posteriori
Pxx_poster = Pxx_priori-K*Pyy*K'; % Eq. 27
Pxx_poster = (Pxx_poster+Pxx_poster')/2;

% Update of state estimation based on information a posteriori
innov = z-y_priori;
x_poster = x_priori+K*innov; % Eq. 26
x_poster = Projection(x_poster, u, Constraints, Pxx_poster); % Eq. 36

end