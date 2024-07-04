function dxdt = ODE_JobsesZymomonas(t,x,u)
% Definition of the system of ordinary differential equations corresponding
% to the ethanol production from zymomonas mobilis fermentation of glucose 
% in a continuous stirred tank reactor (CSTR), according to modeling from 
% Jobses et al. (1993)
%
% Reactions:
% Cyclopentatiene (S) + Cyclopentenol (X) -> Cyclopentanediol (P) + a Cyclopentanediol (X)
% Cyclopentatiene (S) + Cyclopentenol (E) -> b Cyclopentanediol (E);
% Cyclopentatiene (S) + Cyclopentenol (E) -> Cyclopentanediol (P) + a Cyclopentanediol (X)
%
% Inputs:
% t - Time (h)
% x - States
% x(1) - Substrate (glucose) concentration (Cs) [g/L]
% x(2) - Biomass (zymomonas mobilis) concentration (Cx) [g/L]
% x(3) - Artificial concentration to lag the system dynamics (Ce) [g/L]
% x(4) - Product (ethanol) concentration (Cp) [g/L]
% u - Time-invariant inputs (model parameters)
% u(1) - Dilution rate (D) [h^-1]
% u(2) - Concentration of substrate at the inflow (Cs0) [kg/m^3]
% u(3) - Concentration of biomass at the inflow (Cx0) [kg/m^3]
% u(4) - Artificial concentration at the inflow (Ce0) [kg/m^3]
% u(5) - Product concentration at the inflow (Cp0) [kg/m^3]
% u(6) - Rate constant for formation of E biomass [m^6/(kg^2.h)]
% u(7) - Rate constant for formation of E biomass [kg/m^3]
% u(8) - Rate constant for formation of E biomass [kg/m^3]
% u(9) - Monod constrant [kg/m^3]
% u(10) - Maintenance factor based on substrate requirement (ms) [kg/(kg.h)]
% u(11) - Maintenance factor based on production formation (mp) [kg/(kg.h)]
% u(12) - Yield factor of biomass on substrate (Ysx) [kg/kg]
% u(13) - Yield factor of biomass on product (Ypx) [kg/kg]
% u(14) - Maximum specific growth rate (\mu_{max}) [kJ/(kg.K)]
%
% Outputs:
% dxdt - State derivative with respect to time
%
% References:
% Jöbses, I. M. L., Egberts, G. T. C., Luyben, K. C. A. M., & Roels, J. A. 
% (1986). Fermentation kinetics of Zymomonas mobilis at high ethanol concentrations: 
% oscillations in continuous cultures. Biotechnology and bioengineering, 28(6), 868-877.
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024

% Definition of model parameters estimated
D = u(1); % [h^-1];
Cs0 = u(2); % [kg/m^3]

% Definition of model parameters fixed
Cx0 = u(3); % [kg/m^3]
Ce0 = u(4); % [kg/m^3]
Cp0 = u(5); % [kg/m^3]
Ke = u(6); % [m^6/(kg^2*h)]
c1 = u(7); % [kg/m^3]
c2 = u(8); % [kg/m^3]
Ks = u(9); % [kg/(kg.h)]
ms = u(10); % [kg/(kg.h)]
mp = u(11); % [kg/(kg.h)]
Ysx = u(12); % [kg/kg]
Ypx = u(13); % [kg/kg]
mu_max = u(14); % [h^-1]

% ODE system:
% dCs/dt = -mu_max*Cs*Ce/(Ysx*(Ks+Cs)) - ms*Cx + D*(Cs0-Cs)
% dCx/dt = mu_max*Cs*Ce/(Ks+Cs) + D*(Cx0-Cx)
% dCe/dt = Ke*(Cp-c1)*(Cp-c2)*(Cs*Ce/(Ks+Cs)) + D*(Ce0-Ce)
% dCp/dt = mu_max*Cs*Ce/(Ypx*(Ks+Cs)) + mp*Cx + D*(Cp0-Cp)
dxdt(1,:) = -mu_max*x(1,:).*x(3,:)./(Ysx*(Ks+x(1,:))) - ms*x(2,:) + D*(Cs0-x(1,:));
dxdt(2,:) = mu_max*x(1,:).*x(3,:)./(Ks+x(1,:)) + D*(Cx0-x(2,:));
dxdt(3,:) = Ke*(x(4,:)-c1).*(x(4,:)-c2).*(x(1,:).*x(3,:)./(Ks+x(1,:))) + D*(Ce0-x(3,:));
dxdt(4,:) = mu_max*x(1,:).*x(3,:)./(Ypx*(Ks+x(1,:))) + mp*x(2,:) + D*(Cp0-x(4,:));

end

