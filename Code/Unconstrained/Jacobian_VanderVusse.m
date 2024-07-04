function Jacobian = Jacobian_VanderVusse(x,u)
% Definition of the Jacobian from the model proposed by Katt and Engell
% (1993), corresponding to Van der Vusse reactions in a jacketed continuous
% stirred tank reactor (CSTR).
%
% Reactions:
% Cyclopentatiene (A) + Cyclopentenol (B) -> Cyclopentanediol (C)
% 2 Cyclopentatiene (A) -> Dicyclopentadiene (D)
%
% Inputs:
% t - Time (h)
% x - States
% x(1) - Concentration of reactant A in the reactor (Ca) [mol/L]
% x(2) - Concentration of product A in the reactor (Cb) [mol/L]
% x(3) - Temperature of the reactor (T) [K]
% x(4) - Temperature of the reactor coolant (Tk) [K]
% u - Time-invariant inputs (model parameters)
% u(1) - Volumetric flowrate at the reactor [h^-1]
% u(2) - Amount of heat removed by an external heat exchanger (Q_k) [kJ/h]
% u(3) - Concentration of reactant A at the inflow [mol/L]
% u(4) - Arrhenius factor of reaction A->B (k_0_{AB}) [h^-1]
% u(5) - Arrhenius factor of reaction B->C (k_0_{BC}) [h^-1]
% u(6) - Arrhenius factor of reaction A->D (k_0_{AD}) [L/mol.h]
% u(7) - Activation energy of reaction A->B (Ea_{AB}) [kJ/mol]
% u(8) - Activation energy of reaction B->C (Ea_{BC}) [kJ/mol]
% u(9) - Activation energy of reaction A->D (Ea_{AD}) [kJ/mol]
% u(10) - Enthalpy of reaction A->B (dH_{AB}) [kJ/(mol A)]
% u(11) - Enthalpy of reaction B->C (dH_{BC}) [kJ/(mol B)]
% u(12) - Enthalpy of reaction A->D (dH_{AD}) [kJ/(mol A)]
% u(13) - Fluid density [kg/L]
% u(14) - Fluid heat capacity [kJ/(kg.K)]
% u(15) - Coolant heat capacity [kJ/(kg.K)]
% u(16) - Surface of cooling jacket [m^2]
% u(17) - Reactor volume [L]
% u(18) - Coolant mass [kg]
% u(19) - Temperature at the inflow rate [K]
% u(20) - Coolant heat transfer coefficient [kJ/(h.m^2.k)]
%
% Outputs:
% Jacobian - Jacobian of the differential state transition function
%
% References:
% Engell, S., & Klatt, K. U. (1993, June). Nonlinear control of a non-minimum-phase
% CSTR. In 1993 American Control Conference (pp. 2941-2945). IEEE.
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024


% Definition of model parameters estimated
F = u(1); % [h^-1];
Q = u(2); % [kJ/h]
Ca0 = u(3); % [mol/L]

% Definition of model parameters fixed
k1 = u(4); % [h^-1]
k2 = u(5); % [h^-1]
k3 = u(6); % [L/mol.h]
Ea1 = u(7); % [kJ/mol]
Ea2 = u(8); % [kJ/mol]
Ea3 = u(9); % [kJ/mol]
dH1 = u(10); % [kJ/(mol A)]
dH2 = u(11); % [kJ/(mol B)]
dH3 = u(12); % [kJ/(mol A)]
rho = u(13); % [kg/L]
cp = u(14); % [kJ/(kg.K)]
cpk = u(15); % [kJ/(kg.K)]
Ar = u(16); % [m^2]
Vr = u(17); % [L]
mK = u(18); % [kg]
T0 = u(19); % [K]
kw = u(20); % [kJ/(h.m^2.k)]

% Definition of dependent variables
% Reaction rate coefficient
K1 = k1*exp(-Ea1./x(3,:)); % [h^-1]
K2 = k2*exp(-Ea2./x(3,:)); % [h^-1]
K3 = k3*exp(-Ea3./x(3,:)); % [L/mol.h]
% Reaction rate
r1 = K1.*x(1,:); % [mol/h]
r2 = K2.*x(2,:); % [mol/h]
r3 = K3.*x(1,:).^2; % [mol/h]

% ODE system:
% dCa/dt = (F/Vr)*(Ca0-Ca) - r1 - r3;
% dCb/dt = -(F/Vr)*Cb + r1 - r2;
% dT/dt = (T0-T)*(F/Vr) + kw*Ar*(Tk-T)/(rho*Cp*Vr)-...
%          (K1*Ca*dH1+K2*Cb*dH2+K3*Ca^2*dH3)/(rho*Cp);
% dTk/dt = (Qk + kw*Ar*(T-Tk))/(mK*Cp);

Jacobian = [-F/Vr-K1-2*K3*x(1), 0, -(Ea1*r1+Ea3*r3)/(x(3)^2), 0;...
    K1, -F/Vr-K2, (Ea1*r1-Ea2*r2)/(x(3)^2), 0;...
    -1/(rho*cp)*(K1*dH1+2*K3*x(1)*dH3), -1/(rho*cp)*K2*dH2, -F/Vr-(dH1*Ea1*r1+dH2*Ea2*r2+dH3*Ea3*r3)/(rho*cp*x(3)^2)-kw*Ar/(rho*cp*Vr), kw*Ar/(rho*cp*Vr);...
    0, 0, kw*Ar/(mK*cpk), -kw*Ar/(mK*cpk)];

col5 = [F/Vr; 0; 0; 0];
row5 = zeros(1,5);
if numel(x)>4
    Jacobian = [Jacobian, col5; row5];
end
end