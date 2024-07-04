function [Pxx_poster, x_poster, Particle_poster, Particle_W, Particle_priori] =...
    CPF(F, H, Particle_prior, u, z, Q, R, Particle_W_prior, Constraints, Pxx_prior, Jacob_y, varargin)
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
% M. S. Arulampalam, S. Maskell, N. Gordon and T. Clapp, A tutorial on
% particle filters for online nonlinear/non-Gaussian Bayesian tracking, in
% IEEE Transactions on Signal Processing, vol. 50, no. 2, pp. 174-188, Feb.
% 2002, doi: 10.1109/78.978374
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note1: The variables with the prefix "Particle" have calculations for each
% particle propagated based on the column position.
%
% Note2: The variable Particle_priori is returned to speed up computational
% time of a forward-backward filtering.

% Definition of the number of states, measurements and particles
[nx,n_particle] = size(Particle_prior);
ny = numel(z);

% Definition of intermediate variables in the calculations
Q_chol = chol(Q,"lower");
inv_R = mldivide(R,eye(ny));
inv_Q = mldivide(Q,eye(nx));

% Memory allocation for estimations based information a priori
Particle_priori = zeros(nx,n_particle); % State
Particle_Lc = zeros(1,n_particle); % State
Particle_preexp = zeros(1,n_particle); % State
Output_priori = zeros(ny,1); % Measurement

% Checks if x_poster from the unconstrained problem is within the feasible region
D = [];
d = [];
flag_nln = 0;
if isempty(Jacob_y)
    flag_nln = 1;
end
h_poster = Constraints.h(Particle_prior(1:nx,1),u);
if ~isempty(h_poster)
    flag_nln = 1;
    nh = numel(h_poster);
    fun_h = @(var) Constraints.h(var,u);
    g_poster = Constraints.g(Particle_prior(1:nx,1),u);
    if ~isempty(g_poster)
        ng = numel(g_poster);
        fun_g = @(var) Constraints.g(var,u);
    else
        ng = 0;
        fun_g = @(var) [];
    end
else
    g_poster = Constraints.g(Particle_prior(1:nx,1),u);
    if ~isempty(g_poster)
        flag_nln = 1;
        ng = numel(g_poster);
        fun_g = @(var) Constraints.g(var,u);
    else
        ng = 0;
        fun_g = @(var) [];
    end
    fun_h = @(var) [];
    nh = 0;
end
if isempty(Constraints.ubx)
    Constraints.ubx = repmat(1e10,nx,1);
end
if isempty(Constraints.lbx)
    Constraints.lbx = repmat(-1e10,nx,1);
end
Dlin = @(var) [Constraints.Aeq(var,u); -Constraints.Aeq(var,u);...
    Constraints.A(var,u); eye(nx);-eye(nx)];
dlin = @(var) [-Constraints.Beq(var,u); -Constraints.Beq(var,u);...
    Constraints.B(var,u); Constraints.ubx; -Constraints.lbx];

% Definition of linear constraints


Con_check = @(var) [fun_h(var); fun_g(var); Dlin(var)*var]-...
    [zeros(nh+ng,1); dlin(var)];

% Propagation of particles' properties between sampling times at k-1 and k
for i = 1:n_particle % Line 1 from Algorithm 4
    % Propagation of particles based on the dynamic model
    Particle_priori(1:nx,i) = F(Particle_prior(1:nx,i),u)+Q_chol*randn(nx,1); % Line 2 from Algorithm 4

    Particle_Lc(i) = ~any(Con_check(Particle_priori(1:nx,i)) > 1e-10);

    % Estimation of measurements based on the information a priori
    Output_priori(1:ny,i) = H(Particle_priori(1:nx,i)); % Line 3 from Algorithm 4 part 1

    % Calculation of the particles' weights based on the probability density
    % function (PDF)
    Particle_preexp(i) = (-0.5*(z(1:ny,1) - Output_priori(1:ny,i))'*...
        inv_R*(z(1:ny,1) - Output_priori(1:ny,i)));
    % OBS: The current implementations assumes process and measurement noises
    % have normal distributions. Changes at previous equation and at "Q_chol*randn(nx,1)"
    % are required for other distributions.
end

Particle_W_priori = Particle_W_prior'.*exp(Particle_preexp);

% Normalization of particles' weights
Particle_W_chi = Particle_W_priori.*Particle_Lc./sum(Particle_W_priori.*Particle_Lc);


% Definition of state estimation based on the weighted average among the particles
x_chi= zeros(nx,1);
for i = 1:n_particle
    x_chi = x_chi + Particle_W_chi(i)*Particle_priori(1:nx,i);
end

% Definition of tolerances related to the optimization problem
TolX = 1e-6;
TolFun = 1e-6;
TolCon = 1e-8;

if (z-H(x_chi))'*inv_R*(z-H(x_chi)) < 0.05
    Particle_W_priori = Particle_W_chi;
else
    for i = 1:n_particle
        if Particle_W_chi(i) == 0 || isnan(Particle_W_chi(i))
            if flag_nln
                FO_nln = @(x_proj) (x_proj(1:nx) - Particle_priori(1:nx,i))'/Q*(x_proj(1:nx) - Particle_priori(1:nx,i)) +...
                    (z-H(x_proj))'*inv_R*(z-H(x_proj));
                opts = optimoptions('fmincon','Algorithm','interior-point','TolX',TolX,'TolFun',TolFun,...
                    'TolCon',TolCon,'MaxFunEvals',1e5,'MaxIter',1e5, 'Display', 'none');
                Particle_priori(1:nx,i) = fmincon(@(var) FO_nln(var), Particle_priori(1:nx,i), Aw, Bw, [], [], [], [],...
                    @(var) Restr_nln(var, fun_g, fun_h), opts);
            else
                coef_quad = inv_Q + Jacob_y'*inv_R*Jacob_y;
                coef_quad = (coef_quad + coef_quad')/2;
                coef_lin = -inv_Q*Particle_priori(1:nx,i)-Jacob_y'*inv_R*z;
                opts = optimset('Algorithm','interior-point-convex','Display','none','TolX',TolX,...
                    'TolFun', TolFun,'TolCon',TolCon,'MaxIter',1e5);

                Aw = Dlin(Particle_priori(1:nx,i));
                Bw = dlin(Particle_priori(1:nx,i));
                Particle_priori(1:nx,i) = quadprog(coef_quad, coef_lin, Aw,...
                    Bw, [], [], [], [], Particle_priori(1:nx,i),opts);
            end

            % Estimation of measurements based on the information a priori
            Output_priori(1:ny,i) = H(Particle_priori(1:nx,i)); % Line 3 from Algorithm 4 part 1

            % Calculation of the particles' weights based on the probability density
            % function (PDF)
            Particle_preexp(i) = (-0.5*(z(1:ny,1) - Output_priori(1:ny,i))'*...
                inv_R*(z(1:ny,1) - Output_priori(1:ny,i)));

            Particle_W_priori(i) = Particle_W_prior(i)'*((2*pi)^(-ny/2)*det(R)^(-0.5))*...
                exp(-0.5*(z(1:ny,1) - Output_priori(1:ny,i))'*...
                inv_R*(z(1:ny,1) - Output_priori(1:ny,i)));

        end
    end

    Particle_W_priori = Particle_W_prior'.*exp(Particle_preexp);
    it = 1;
    while ~any(Particle_W_priori>eps) && it<100
        Particle_preexp = Particle_preexp/exp(1);
        Particle_W_priori = Particle_W_prior'.*exp(Particle_preexp);
        it = it + 1;
    end
    Particle_W_priori = Particle_W_priori./sum(Particle_W_priori); % Lines 6-8 from Algorithm 4
end

% Definition of effective sample size
Neff = 1/sum(Particle_W_priori.^2); % Eq. 51

% Definition of a threshold condition to resample particles
N_T = n_particle/10; % Arbitrary definition

if Neff < N_T || true
    % OBS: Comment "|| true" at the previous line to change implementation
    % of Algorithm 4 to Algorithm 3 from Arulampalam et al. (2002)

    % Calculation of cumulative distribution function (CDF) of particle's weights
    CDF = cumsum(Particle_W_priori); % Lines 1-4 from Algorithm 2

    % Definition of a uniform distribution with respect to the particle size
    u_dist = (0:1/n_particle:1) + rand(1)/n_particle; % Lines 5 and 6 from Algorithm 2
    u_dist(end) = 1; % Line 6 from Algorithm 2

    i = 1;
    Particle_poster = zeros(nx,n_particle);

    % Resample the particles to adjust their weights to a uniform distribution
    for j = 1:n_particle % Line 7 from Algorithm 2
        % Move along the particles until find a CDF in agreement with the uniform distribution
        while u_dist(j)>CDF(i) % Line 8-11 from Algorithm 2
            i = i+1;
        end
        % Update the particles' estimation
        Particle_poster(:,j) = Particle_priori(:,i); % Line 12 from Algorithm 2
    end % Line 6 from Algorithm 2

    % Update the particles' weight
    Particle_W = repmat(1/n_particle, 1, n_particle); % Line 13 from Algorithm 2
    % OBS: Line 14 from Algorithm 2 is neglected because it refers to
    % parentage tracking in the resampling algorithm.
else
    Particle_poster = Particle_priori;
    Particle_W = Particle_W_priori;
end

% Definition of state estimation based on the weighted average among the particles
x_poster= zeros(nx,1);
for i = 1:n_particle
    x_poster = x_poster + Particle_W(i)*Particle_poster(:,i);
end

% Definition of the average covariance of the state estimated within the particles resampled
Pxx_poster = zeros(nx,nx);
den = 0;
for i = 1:n_particle
    Pxx_poster = Pxx_poster+Particle_W(i)*(Particle_poster(1:nx,i)-x_poster)*...
        (Particle_poster(1:nx,i)-x_poster)'; % Eq. 49
    den = den + Particle_W(i)^2;
end
Pxx_poster = Pxx_poster/(1-den);
end
function [c, ceq] = Restr_nln(w, fun_g, fun_h)
% Definition of nonlinear constraints according to the input format from MATLAB
%
% Inputs:
% w - Process noises
% fun_g - Nonlinear inequality constraints according to g(w) <= 0
% fun_h - Nonlinear equality constraints according to h(w) = 0
%
% Outputs:
% c - Nonlinear inequality constraints c <= 0
% ceq - Nonlinear inequality constraints ceq <= 0
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024

% Definition of the inputs and outputs sizes
c = fun_g(w);
ceq = fun_h(w);

end