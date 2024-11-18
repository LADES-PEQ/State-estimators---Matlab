function [Pxx_filt, x_filt] = MHE(F, H , X_vector, U_vector, Pxx_vector,...
    z_vector, Q_vector, R_vector, Constraints, Jacob_x, Jacob_y, varargin)
% Calculation of the moving horizon estimation (MHE) based on the methodology
% proposed by Rao et al. (2003)
%
% Inputs:
% F - State transition function
% H - Measurement function
% X_vector - State estimations between k-N-1 and k-1
% U_vector - Set of time-invariant variables between k-N-1 and k-1
% Pxx_vector - State error covariance matrices between k-N-1 and k-1
% z_vector - Measurements from the system between k-N-1 and k
% Q_vector - Covariance matrices of the process noises between k-N-1 and k-1
% R_vector - Covariance matrices of the observation noises between k-N-1 and k
% Constraints - Structure comprising the set of model constraints
% Jacob_x - Numerical calculation of the state transition matrix
% Jacob_y - Measurement matrix
%
% Outputs:
% Pxx_filt - State error covariance matrices between k-N-1 and k
% x_filt - State estimations between k-N-1 and k
%
% References:
% Rao, C. V., Rawlings, J. B. & Mayne, D. Q. Constrained State Estimation
% for Nonlinear Discrete-Time Systems: Stability and Moving Horizon
% Approximations. IEEE Transactions on Autom. Control. 48, 246–258,
% doi: 10.1109/TAC.2002.808470 (2003).
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024

% Definition of the moving horizon and the number of states, measurements
% and model input parameters
[nx,N] = size(X_vector);
ny = size(z_vector,1);
nu = size(U_vector,1);

% Reshape of input variables in column vectors
X_col = reshape(X_vector(1:nx,2:N), nx*(N-1),1);
z_col = reshape(z_vector(1:ny,2:N+1),N*ny,1);
X_col = [X_col;X_vector(1:nx,N)];

% Reshape of input variables in diagonal matrices
R_col = [];
Q_col = [];
for i = 2:N
    R_col = blkdiag(R_col, R_vector(1:ny,1:ny,i));
    Q_col = blkdiag(Q_col, Q_vector(1:nx,1:nx,i-1));
end
R_col = blkdiag(R_col, R_vector(1:ny,1:ny,N+1));

% Propagation of states based on the dynamic model for time k-N
x_priori = F(X_vector(1:nx,1), U_vector(1:nu,1));

% Definition of constraints for the column vectors
% Linear inequality constraints in the form A*x_col <= B
A = [];
for i = 1:N
    A = blkdiag(A,Constraints.A);
end
B = repmat(Constraints.B,N,1);
% Linear equality constraints in the form Aeq*x_col <= Beq
Aeq = [];
for i = 1:N
    Aeq = blkdiag(Aeq,Constraints.Aeq);
end
Beq = repmat(Constraints.Beq,N,1);
% Boundary constraints
ubx = repmat(Constraints.ubx,N,1);
lbx = repmat(Constraints.lbx,N,1);
% Nonlinear constraints
if ~isempty(Constraints.h)
    if ~isempty(Constraints.g)
        fun_g = @(x) Constraints.g(x(1:nx));
    else
        fun_g = @(x) [];
    end
    fun_h = @(x) Constraints.h(x);
else
    fun_h = @(x) [];
    if ~isempty(Constraints.g)
        fun_g = @(x) Constraints.g(x(1:nx));
    else
        fun_g = @(x) [];
    end
end

% Definition of tolerances related to the optimization problem
TolX = 1e-10;
TolFun = 1e-10;
TolCon = 1e-10;

% Note: Rao et al. (2003) defined the optimization problem with respect to 
% the state at k-N and the process noises in the moving horizon. The current
% implementation solves the optimization problem directly for the state
% estimations, which eases the constraint definition and provides more
% comprehensive outputs. 

% Solves the optimization problem
opts = optimoptions('fmincon','Algorithm','interior-point','Display','none',...
    'TolX',TolX,'TolFun',TolFun,'TolCon',TolCon,'MaxFunEvals',1e5,'MaxIter',...
    1e5,'UseParallel', false,'FiniteDifferenceType','central',...
    'SubproblemAlgorithm','factorization');
aux = fmincon(@(var) FO(F, H, var, U_vector, Pxx_vector, z_col, Q_col, R_col,...
    x_priori, nx, ny, nu, N), X_col, A, B, Aeq, Beq, lbx, ubx,...
    @(var) Restr_nln(var, fun_g, fun_h, nx, N), opts);

% Update of the constrained state estimations
x_filt = [X_vector(1:nx,1), reshape(aux,nx,N)];

% Calculation of the state transition matrix
phi = Jacob_x(x_filt(1:nx,N), U_vector(1:nu,N));

% Update of state error covariance matrix based on information a posteriori
Pxx_filt = zeros(nx,nx,N+1);
Pxx_filt(1:nx,1:nx,1:N) = Pxx_vector;
Pxx_filt(:,:,N+1) = phi*Pxx_vector(:,:,N)*phi'-phi*Pxx_vector(:,:,N)*Jacob_y'/(Jacob_y*Pxx_vector(:,:,N)*Jacob_y'+R_vector(1:ny,1:ny,N+1))*Jacob_y*Pxx_vector(:,:,N)*phi'+Q_vector(1:nx,1:nx,N);
end
function FO = FO(F, H, X_col, U_vector, Pxx_vector, z_col, Q_col, R_col,...
    x_pri, nx, ny, nu, N, varargin)
% Objective function of the moving horizon estimation
%
% Inputs:
% F - State transition function
% H - Measurement function
% X_col - State estimations between k-N-1 and k-1
% U_vector - Set of time-invariant variables between k-N-1 and k-1
% Pxx_vector - State error covariance matrices between k-N-1 and k-1
% z_vector - Measurements from the system between k-N-1 and k
% Q_vector - Covariance matrices of the process noises between k-N-1 and k-1
% R_vector - Covariance matrices of the observation noises between k-N-1 and k
% x_pri - A priori state estimation at k-N
% nx - Number of states
% ny - Number of measurements
% nu - Number of time-invariant variables
% N - Size of the moving horizon
%
% Outputs:
% FO - Objective function
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024

x_priori = zeros(N*nx,1);
x_priori(1:nx) = x_pri;
Y_col = zeros(N*ny,1);
Y_col(1:ny) = H(X_col(1:nx));
for i = 1:N-1
    x_priori(i*nx+1:(i+1)*nx) = F(X_col((i-1)*nx+1:i*nx),U_vector(1:nu,i+1));
    Y_col(i*ny+1:(i+1)*ny) = H(X_col(i*nx+1:(i+1)*nx));
end
FO = (z_col - Y_col)'/R_col*(z_col - Y_col) + (X_col(1:nx)-x_priori(1:nx))'/...
    Pxx_vector(1:nx,1:nx,1)*(X_col(1:nx)-x_priori(1:nx));
if N>1
    FO = FO + (X_col(nx+1:N*nx) - x_priori(nx+1:N*nx))'/Q_col*...
        (X_col(nx+1:N*nx) - x_priori(nx+1:N*nx));
end
end
function [c, ceq] = Restr_nln(x, fun_g, fun_h, nx, N)
% Definition of nonlinear constraints according to the input format from MATLAB
%
% Inputs:
% x - State
% fun_g - Nonlinear inequality constraints according to g(w) <= 0
% fun_h - Nonlinear equality constraints according to h(w) = 0
% nx - Number of states
% N - Size of the moving horizon
%
% Outputs:
% c - Nonlinear inequality constraints c <= 0
% ceq - Nonlinear inequality constraints ceq <= 0
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024

c = [];
ceq = [];
for i = 1:N
    c = [c; fun_g(x((i-1)*nx+1:i*nx))];
    ceq = [ceq; fun_h(x((i-1)*nx+1:i*nx))];
end
end