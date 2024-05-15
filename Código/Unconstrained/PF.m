function [Pxx_poster, x_poster, Particle_poster, Particle_W, Particle_priori] =...
    PF(F, H, Particle_prior, u, z, Q, R, Particle_W_prior, varargin)
% Calculation of the sequential importance resampling (SIR) particle
% filter, also known as the bootstrap filter, based on the methodology 
% proposed by Arulampalam et al. (2002)
%
% Inputs:
% F - State transition function
% H - Measurement function
% Particle_prior - State estimations at time k-1 for each particle
% u - Set of time-invariant variables between sampling times k-1 and k
% z - System measurements at time k
% Q - Covariance matrix of the process noises
% R - Covariance matrix of the observation noises
% Particle_W_prior - Particles' weights calculated at time k-1
%
% Outputs:
% Pxx_poster - State error covariance matrix at time k
% x_poster - State estimation at time k
% Particle_poster - State estimations at time k for each particle
% Particle_W - Particles' weights calculated at time k
% Particle_priori - A priori state estimation for each particle at time k
%
% References:
% Arulampalam, M. S., Maskell, S., Gordon, N., & Clapp, T. (2002). A tutorial
% on particle filters for online nonlinear/non-Gaussian Bayesian tracking. 
% IEEE Transactions on signal processing, 50(2), 174-188.
%
% Sarkka, S. (2013). Bayesian filtering and smoothing. Cambridge University Press
%
% Shao, X., Huang, B., & Lee, J. M. (2010). Constrained Bayesian state 
% estimation–A comparative study and a new particle filter based approach. 
% Journal of Process Control, 20(2), 143-157.
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The variables with the prefix "Particle" have calculations for each 
% particle propagated based on the column position.
%
% Note2: Equation numbers refer to Sarkka (2013).
%
% Note3: The variable Particle_priori is returned to speed up computational
% time of a forward-backward filtering.
%
% Note4: Arulampalam et al. (2002) is recommended for better understanding 
% of the algorithm


% Definition of the number of states, measurements and particles
[nx,n_particle] = size(Particle_prior);
ny = numel(z);

% Definition of intermediate variables in the calculations
Q_chol = chol(Q,"lower");
invR = mldivide(R,eye(ny));

% Memory allocation for estimations based information a priori
Particle_priori = zeros(nx,n_particle); % State
Output_priori = zeros(ny,1); % Measurement

% Propagation of particles' properties between sampling times at k-1 and k 
for i = 1:n_particle % Line 1 from Algorithm 4
    % Propagation of particles based on the dynamic model
    Particle_priori(1:nx,i) = F(Particle_prior(1:nx,i),u)+Q_chol*randn(nx,1); % Eq. 7.24
    
    % Estimation of measurements based on the information a priori
    Output_priori(1:ny,i) = H(Particle_priori(1:nx,i)); % Eq. 7.25 part 1

    % Calculation of the particles' weights based on the probability density
    % function (PDF)
    Particle_W_prior(i) = Particle_W_prior(i)'*((2*pi)^(-ny/2)*det(R)^(-0.5))*...
        exp(-0.5*(z(1:ny,1) - Output_priori(1:ny,i))'*...
        invR*(z(1:ny,1) - Output_priori(1:ny,i))); % Eq. 7.25 part 2
    % OBS: The current implementations assumes process and measurement noises
    % have normal distributions. Changes at previous equation and at "Q_chol*randn(nx,1)"
    % are required for other distributions.
end

% Calculation of total particles' weight
Wt = sum(Particle_W_prior); % Eq. 7.25 part 3

% Normalization of particles' weights
Particle_W_prior = Particle_W_prior./Wt; % Eq. 7.25 part 4

% Definition of effective sample size
Neff = 1/sum(Particle_W_prior.^2); % Eq. 7.27

% Definition of a threshold condition to resample particles
N_T = n_particle/10; % Arbitrary definition

if Neff < N_T || true
    % OBS: Comment "|| true" at the previous line to change implementation 
    % of Algorithm 4 to Algorithm 3 from Arulampalam et al. (2002)

    % Calculation of cumulative distribution function (CDF) of particle's weights
    CDF = cumsum(Particle_W_prior); % Lines 1-4 of Algorithm 2 from Arulampalam et al. (2002)

    % Definition of a uniform distribution with respect to the particle size
    u_dist = (0:1/n_particle:1) + rand(1)/n_particle; % Lines 5 and 6 of Algorithm 2 from Arulampalam et al. (2002)
    u_dist(end) = 1; % Line 6 of Algorithm 2 from Arulampalam et al. (2002)

    i = 1;
    Particle_poster = zeros(nx,n_particle);

    % Resample the particles to adjust their weights to a uniform distribution
    for j = 1:n_particle % Line 7 of Algorithm 2 from Arulampalam et al. (2002)   
        % Move along the particles until find a CDF in agreement with the uniform distribution
        while u_dist(j)>CDF(i) % Line 8-11 from Algorithm 2
            i = i+1;
        end
        % Update the particles' estimation
        Particle_poster(:,j) = Particle_priori(:,i); % Line 12 of Algorithm 2 from Arulampalam et al. (2002)
    end % Line 6 of Algorithm 2 from Arulampalam et al. (2002)

    % Update the particles' weight
    Particle_W = repmat(1/n_particle, 1, n_particle); % Line 13 of Algorithm 2 from Arulampalam et al. (2002)
    % OBS: Line 14 is neglected because it refers to parentage tracking in the resampling algorithm.
else
    Particle_poster = Particle_priori;
    Particle_W = Particle_W_priori;
end

% Definition of state estimation based on the weighted average among the particles
x_poster= zeros(nx,1);
for i = 1:n_particle
    x_poster = x_poster + Particle_W(i)*Particle_poster(:,i); % Eq. 24 part 1 from Shao et al. (2010)
end

% Definition of the average covariance of the state estimated within the particles resampled
Pxx_poster = zeros(nx,nx);
den = 0;
for i = 1:n_particle
    Pxx_poster = Pxx_poster+Particle_W(i)*(Particle_poster(1:nx,i)-x_poster)*...
        (Particle_poster(1:nx,i)-x_poster)'; % Eq. 24 part 2 from Shao et al. (2010)
    den = den + Particle_W(i)^2; % Eq. 24 part 3 from Shao et al. (2010)
end
Pxx_poster = Pxx_poster/(1-den); % Eq. 24 part 4 from Shao et al. (2010)

end