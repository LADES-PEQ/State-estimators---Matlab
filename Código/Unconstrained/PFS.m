function [Pxx_filt, x_filt, Particle_filt, Particle_Wf, Particle_priori] = PFS(F,...
    H, Particle_vector, U_vector, z_vector, Q_vector, R_vector, Particle_W,...
    Particle_priori, varargin)
% Calculation of the particle filter and smoother (PF&S) for a forward-backward
% filtering approach, in which the filtering and smoothing estimations follow
% the algorithms proposed by Arulampalam et al. (2002) and Doucet and Johansen
% (2009), respectively.
%
% Inputs:
% F - State transition function
% H - Measurement function
% Particle_vector - State estimations between k-N-1 and k-1 for each particle
% U_vector - Set of time-invariant variables between k-N-1 and k-1
% z_vector - Measurements from the system at time k
% Q_vector - Covariance matrix of the process noises
% R_vector - Covariance matrix of the observation noises
% Particle_W - Particles' weights calculated between k-N and k-1
% Particle_priori - A priori estimations for each particle between k-N and k-1
%
% Outputs:
% Pxx_filt - State error covariance matrices between k-N-1 and k
% x_filt - State estimations between k-N-1 and k
% Particle_filt - State estimations between k-N-1 and k for each particle
% Particle_Wf - Particles' weights calculated between k-N-1 and k
% Particle_priori - A priori estimations for each particle between k-N and k
%
% References:
% M. S. Arulampalam, S. Maskell, N. Gordon and T. Clapp, A tutorial on
% particle filters for online nonlinear/non-Gaussian Bayesian tracking, in
% IEEE Transactions on Signal Processing, vol. 50, no. 2, pp. 174-188, Feb.
% 2002, doi: 10.1109/78.978374
%
% Doucet, A., & Johansen, A. M. (2009). A tutorial on particle filtering and
% smoothing: Fifteen years later. Handbook of nonlinear filtering, 12(656-704), 3.
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: Equation numbers refer to Doucet and Johansen (2009).
%
% Note2: The variables with the prefix "Particle" have calculations for each
% ensemble member propagated based on the column position.
%
% Note3: The execution time of the forward-backward filtering is extremely 
% slower for larger number of particles. In addition, model accuracy is more
% sensitive to particle size than the smoothing horizon. Hence, a practical
% tuning of the observer must focus on the number of particles. The tuning 
% of a non-zero smoothing horizon might be relevant only for theoretical works
% and rigorous fine-tuning.

% Definition of the smoothing horizon and the number of states,
% measurements and particles
[nx,n_particle,N] = size(Particle_vector);
ny = size(z_vector,1);
nu = size(U_vector,1);

% Memory allocation for output variables of the smoothing estimations
x_smo = zeros(nx,N);
Particle_Ws = zeros(n_particle,N);
Pxx_smo = zeros(nx,nx,N);

% Definition of initial conditions for the smoothing estimations based on
% estimations a posteriori at time k-1
Particle_Ws(1:n_particle,N) = Particle_W(1:n_particle,N);

% Smoothing estimations from k-2 to k-N-1
for i = 1:N-1
    % Calculation of smoothed particles' weights
    inv_Q = mldivide(eye(nx),Q_vector(1:nx,1:nx,N-i));
    for ii = 1:n_particle
        sum_num = 0;
        for j = 1:n_particle
            sum_dem = 0;
            for l = 1:n_particle
                sum_dem = sum_dem + Particle_W(l,N-i)*...
                    ((2*pi)^(-nx/2)*det(Q_vector(1:nx,1:nx,N-i))^(-0.5))*...
                    exp(-0.5*(Particle_vector(1:nx,j,N-i+1) - Particle_priori(1:nx,l,N-i))'...
                    *inv_Q*(Particle_vector(1:nx,j,N-i+1) - Particle_priori(1:nx,l,N-i)));
            end
            sum_num = sum_num + (Particle_Ws(j,N-i+1)/sum_dem)*...
                ((2*pi)^(-nx/2)*det(Q_vector(1:nx,1:nx,N-i))^(-0.5))*...
                exp(-0.5*(Particle_vector(1:nx,j,N-i+1) - Particle_priori(1:nx,ii,N-i))'*...
                inv_Q*(Particle_vector(1:nx,j,N-i+1) - Particle_priori(1:nx,ii,N-i)));
        end
        Particle_Ws(ii,N-i) = Particle_W(ii,N-i)*sum_num;
    end
    Particle_Ws(1:n_particle,N-i) = Particle_Ws(1:n_particle ,N-i)./sum(Particle_Ws(1:n_particle ,N-i));

    % Estimation of the smoothed counterparts of the states (x_smo), and 
    % the state error covariance matrix (Pxx_smo)
    for j = 1:n_particle
        x_smo(1:nx,N-i) = x_smo(1:nx,N-i) + Particle_Ws(j,N-i)*Particle_vector(1:nx,j,N-i);
    end
    for m = 1:n_particle
        Pxx_smo(1:nx,1:nx,N-i) = Pxx_smo(1:nx,1:nx,N-i)+Particle_Ws(j,N-i)*...
            (Particle_vector(1:nx,m,N-i)-x_smo(1:nx,N-i))*...
            (Particle_vector(1:nx,m,N-i)-x_smo(1:nx,N-i))'; % Eq. 49
    end

end

% Memory allocation for output variables of the forward filtering pass
x_filt = zeros(nx,N+1);
Particle_filt = zeros(nx,n_particle,N+1);
Particle_Wf = zeros(n_particle,N+1);
Particle_priori = zeros(nx,n_particle,N+1);
Pxx_filt = zeros(nx,nx,N+1);

% Definition of the initial condition of the forward filtering pass by the
% smoothed estimation at k-N-1
x_filt(1:nx,1) = x_smo(1:nx,1);
Particle_filt(1:nx,1:n_particle,1) = Particle_vector(1:nx,1:n_particle,1);
Particle_Wf(1:n_particle,1) = Particle_Ws(1:n_particle,1);
Pxx_filt(1:nx,1:nx,1) = Pxx_smo(1:nx,1:nx,1);

% Filtering estimations from k-N to k
for i = 1:N
    [Pxx_filt(1:nx,1:nx,i+1), x_filt(1:nx,i+1), Particle_filt(1:nx,1:n_particle,i+1),...
        Particle_Wf(1:n_particle,i+1), Particle_priori(1:nx,1:n_particle,i)] =...
        PF(F, H, Particle_filt(1:nx,1:n_particle,i), U_vector(1:nu,i),...
        z_vector(1:ny,i+1), Q_vector(1:nx,1:nx,i), R_vector(1:ny,1:ny,i+1),...
        Particle_Wf(1:n_particle,i));
end

end