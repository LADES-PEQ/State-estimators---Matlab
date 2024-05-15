function Results = Plotting_Est(Plant, Results, n_sim_scen, N_smo, N_points, dir_save)
% Function to reorganize and plot the simulation results for validation
%
% Inputs:
% Plant - Structure comprising the dynamics of the system
% Results - Structure comprising the simulation results
% n_sim_scen - Number of simulations per scenario
% N_smo - Smoothing horizon
% N_points - Number of state points calculated at each sampling time
% dir_save - Data directory
%
% Outputs:
% Results - Structure comprising the simulation results
% Results.X_sim - Real-time state estimations within the analysis period
% Results.Y_sim - Real-time measurement predictions within the analysis period
% Results.Pxx_sim - Real-time state error covariance matrices within the analysis period
% Results.X_vector - Moving horizon estimations within the analysis period
% Results.Pxx_vector - Moving horizon state error covariance matrices within the analysis period
% Results.Esf_vector - Observer execution time within the analysis period
% Results.X_mean - Average state estimations from the observers for each simulated scenario
% Results.Y_mean - Average measurement predictions from the observers for each simulated scenario
% Results.Esf_mean - Average execution time of observers for each simulated scenario
% Results.flag_est - Observer algorithm implemented in each simulated scenario
% Results.N_smo - Smoothing horizon for each simulated scenario
% Results.N_points - Number of state points calculated in the observer for each simulated scenario
% Results.MSE_x - Mean squared error (MSE) of state estimations for each simulated scenario
% Results.MAPE_x - Mean absolute percentage error (MAPE) of state estimations for each simulated scenario
% Results.MSE_y - MSE of measurement predictions for each simulated scenario
% Results.MAPE_y - MAPE of measurement predictions for each simulated scenario
% Results.LogLikelihood - Average state-dependent term of the log-likehood
% from the observers for each simulated scenario
%
% Programmed by:
% Daniel Martins Silva (dmsilva@peq.coppe.ufrj.br)
% Universidade Federal do Rio de Janeiro, 2024
%
% Note: The function L(x) present in some plots refer independent terms in
% log-likehood with respect to the state estimations, such that log-likehood = -L(x) + Constant.
close all

% Definition of the input sizes
[~,N_k,n_sim] = size(Results{1}.X_sim);
ny = size(Results{1}.Y_sim,1);
n_scen = (n_sim-1)/n_sim_scen;
n_config = 8; % Number of observer configurations evaluated for each simulated scenario
n_case = numel(Results); % Number of case studies

for flag_ode = 1:n_case
    nest = size(Results{flag_ode}.X_sim,1);

    % Memory allocation of remaining simulation outputs
    Results{flag_ode}.X_mean = zeros(1,n_scen);
    Results{flag_ode}.Y_mean = zeros(1,n_scen);
    Results{flag_ode}.Esf_mean = zeros(1,n_scen);
    Results{flag_ode}.flag_est = zeros(1,n_scen);
    Results{flag_ode}.N_smo = zeros(1,n_scen);
    Results{flag_ode}.N_points = zeros(1,n_scen);
    Results{flag_ode}.MSE_x = zeros(nest,n_scen);
    Results{flag_ode}.MAPE_x = zeros(nest,n_scen);
    Results{flag_ode}.MSE_y = zeros(ny,n_scen);
    Results{flag_ode}.MAPE_y = zeros(ny,n_scen);
    Results{flag_ode}.LogLikelihood = zeros(1,n_scen);

    % Calculation of inverse matrices
    inv_Q = mldivide(Results{flag_ode}.Q,eye(nest));
    inv_R = mldivide(Results{flag_ode}.R,eye(ny));

    % Calculation and storage of remaining simulation outputs
    for m = 1:n_scen
        Results{flag_ode}.Esf_mean(m) = mean(mean(Results{flag_ode}.Esf_vector(1,2:N_k,(m-1)*n_sim_scen+2:m*n_sim_scen+1)));
        Results{flag_ode}.flag_est(1,m) = N_smo((m-1)*n_sim_scen+2);
        Results{flag_ode}.N_smo(1,m) = N_smo((m-1)*n_sim_scen+2);
        Results{flag_ode}.N_points(1,m) = N_points((m-1)*n_sim_scen+2);
        for k = 1:N_k
            LogLikelihood = zeros(1,n_sim_scen);
            for j = 1:n_sim_scen
                LogLikelihood(1,j) = LogLikelihood(1,j)+(Plant{flag_ode}.Z(1:ny,k)-...
                    Results{flag_ode}.Y_sim(1:ny,k,(m-1)*n_sim_scen+1+j))'*...
                    inv_R*(Plant{flag_ode}.Z(1:ny,k)-...
                    Results{flag_ode}.Y_sim(1:ny,k,(m-1)*n_sim_scen+1+j));
                LogLikelihood(1,j) = LogLikelihood(1,j)+(Plant{flag_ode}.X(1:nest,k)-...
                    Results{flag_ode}.X_sim(1:nest,k,(m-1)*n_sim_scen+1+j))'*...
                    inv_Q*(Plant{flag_ode}.X(1:nest,k)-Results{flag_ode}.X_sim(1:nest,k,(m-1)*n_sim_scen+1+j));
            end
            Results{flag_ode}.LogLikelihood(1,m) = mean(LogLikelihood(1,1:n_sim_scen));

            for j = 1:nest
                Results{flag_ode}.X_mean(j,k,m) = mean(Results{flag_ode}.X_sim(j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1));
                Pxx_aux = mean(Results{flag_ode}.Pxx_sim(j,j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1));
                Results{flag_ode}.CI_max_mean(j,k,m) = Results{flag_ode}.X_mean(j,k,m) + 1.96*Pxx_aux^0.5;
                Results{flag_ode}.CI_min_mean(j,k,m) = Results{flag_ode}.X_mean(j,k,m) - 1.96*Pxx_aux^0.5;
                Results{flag_ode}.MSE_x(j,m) = Results{flag_ode}.MSE_x(j,m) + (1/(n_sim_scen*N_k))*sum(Results{flag_ode}.X_sim(j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1)-Plant{flag_ode}.X(j,k)).^2;
                Results{flag_ode}.MAPE_x(j,m) = Results{flag_ode}.MAPE_x(j,m) + (1/(n_sim_scen*N_k))*sum(abs((Results{flag_ode}.X_sim(j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1)-Plant{flag_ode}.X(j,k))/Plant{flag_ode}.X(j,k)));
            end
            for j = 1:ny
                Results{flag_ode}.Y_mean(j,k,m) = mean(Results{flag_ode}.Y_sim(j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1));
                Results{flag_ode}.MSE_y(j,m) = Results{flag_ode}.MSE_y(j,m) + (1/(n_sim_scen*N_k))*sum(Results{flag_ode}.Y_sim(j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1)-Plant{flag_ode}.Z(j,k)).^2;
                Results{flag_ode}.MAPE_y(j,m) = Results{flag_ode}.MAPE_y(j,m) + (1/(n_sim_scen*N_k))*sum(abs((Results{flag_ode}.Y_sim(j,k,(m-1)*n_sim_scen+2:m*n_sim_scen+1)-Plant{flag_ode}.Z(j,k))/Plant{flag_ode}.Z(j,k)));
            end
        end
    end
end

% Plotting the simulations results under forward-backward filtering
aux = figure;
for flag_ode = 1:n_case
    aux2 = subplot(3,2,(flag_ode-1)*2+1);
    bar(0:n_config-1,reshape(Results{flag_ode}.LogLikelihood(1:n_config*3),n_config,3));
    xlabel("Smoothing horizon")
    ylabel('$\mathcal{L} (x)$', 'Interpreter','latex');
    axis tight
    aux2.Children(3).FaceColor = [1 0 0];
    aux2.Children(2).FaceColor = [0 0 1];
    aux2.Children(1).FaceColor = [0.5 0.5 0.5];
    aux2.Children(3).DisplayName = "EKF&S";
    aux2.Children(2).DisplayName = "UKF&S";
    aux2.Children(1).DisplayName = "CKF&S";
    aux2.OuterPosition(1) = 0;
    aux2.OuterPosition(2) = 1-flag_ode/3;
    aux2.OuterPosition(3) = 0.5;
    aux2.OuterPosition(4) = 1/3;
    for i = 1:n_config
        aux2.XTickLabel{i} = Results{flag_ode}.N_smo(i);
    end

    aux2 = subplot(3,2,flag_ode*2);
    bar(0:n_config-1,reshape(Results{flag_ode}.Esf_mean(1:n_config*3),n_config,3));
    xlabel("Smoothing horizon")
    ylabel('$T_{comp}$ (s)', 'Interpreter','latex')
    axis tight
    aux2.Children(3).FaceColor = [1 0 0];
    aux2.Children(2).FaceColor = [0 0 1];
    aux2.Children(1).FaceColor = [0.5 0.5 0.5];
    aux2.Children(3).DisplayName = "EKF&S";
    aux2.Children(2).DisplayName = "UKF&S";
    aux2.Children(1).DisplayName = "CKF&S";
    aux2.OuterPosition(1) = 0.5;
    aux2.OuterPosition(2) = 1-flag_ode/3;
    aux2.OuterPosition(3) = 0.5;
    aux2.OuterPosition(4) = 1/3;
    for i = 1:n_config
        aux2.XTickLabel{i} = Results{flag_ode}.N_smo(i);
    end
    aux2.XTickLabelRotation = 0;

    if flag_ode == n_case
        legend("Location", "best");
    end
end
aux.PaperUnits = 'centimeters';
aux.PaperPositionMode='manual';
aux.PaperSize = [14.8 13];
aux.PaperPosition = [0 0 14.8 13];
aux.PaperPositionMode='manual';
print(strcat(dir_save,"Unconst_Filt_Smooth"),"-djpeg","-r300", '-image');

% Plotting the simulations results subject to PF or EnKF estimations
aux = figure;
for flag_ode = 1:n_case
    aux2 = subplot(3,2,(flag_ode-1)*2+1);
    bar(0:n_config-1,reshape(Results{flag_ode}.LogLikelihood(n_config*3+1:n_config*5),n_config,2));
    xlabel("Number of particles")
    ylabel('$\mathcal{L} (x)$', 'Interpreter','latex');
    axis tight
    aux2.Children(2).FaceColor = [1 0.5 0];
    aux2.Children(1).FaceColor = [0 0.5 0.5];
    aux2.Children(2).DisplayName = "PF";
    aux2.Children(1).DisplayName = "EnKF";
    aux2.OuterPosition(1) = 0;
    aux2.OuterPosition(2) = 1-flag_ode/3;
    aux2.OuterPosition(3) = 0.5;
    aux2.OuterPosition(4) = 1/3;
    for i = 1:n_config
        aux2.XTickLabel{i} = Results{flag_ode}.N_points(n_config*3+i);
    end
    aux2.XTickLabelRotation = 0;

    aux2 = subplot(3,2,flag_ode*2);
    bar(0:n_config-1,reshape(Results{flag_ode}.Esf_mean(n_config*3+1:n_config*5),n_config,2));
    xlabel("Number of particles")
    ylabel('$T_{comp}$ (s)', 'Interpreter','latex')
    axis tight
    aux2.Children(2).FaceColor = [1 0.5 0];
    aux2.Children(1).FaceColor = [0 0.5 0.5];
    aux2.Children(2).DisplayName = "PF";
    aux2.Children(1).DisplayName = "EnKF";
    aux2.OuterPosition(1) = 0.5;
    aux2.OuterPosition(2) = 1-flag_ode/3;
    aux2.OuterPosition(3) = 0.5;
    aux2.OuterPosition(4) = 1/3;
    for i = 1:n_config
        aux2.XTickLabel{i} = Results{flag_ode}.N_points(n_config*3+i);
    end
    aux2.XTickLabelRotation = 0;
    if flag_ode == n_case
        legend("Location", "best");
    end
end
aux.PaperUnits = 'centimeters';
aux.PaperPositionMode='manual';
aux.PaperSize = [14.8 13];
aux.PaperPosition = [0 0 14.8 13];
aux.PaperPositionMode='manual';
print(strcat(dir_save,"Unconst_Filt_Part"),"-djpeg","-r300", '-image');

% Definition of the best configuration simulated for each observer
Best = zeros(n_case, 5);
Best_pos = zeros(n_case, 5);
for flag_ode = 1:n_case
    for i = 1:5
        [Best(flag_ode,i), Best_pos(flag_ode,i)] = min(Results{flag_ode}.LogLikelihood((i-1)*n_config+1:i*n_config));
    end
end

% Plotting the best simulations results under forward filtering
aux = figure;
aux2 = subplot(2,1,1);
LogLikelihood = [Results{1}.LogLikelihood([1,n_config+1, 2*n_config+1,...
    4*n_config, 5*n_config]); Results{2}.LogLikelihood([1,n_config+1, 2*n_config+1,...
    4*n_config, 5*n_config]); Results{3}.LogLikelihood([1,n_config+1, 2*n_config+1,...
    4*n_config, 5*n_config])];

bar(1:3,LogLikelihood);
xlabel("Simulation scenario")
ylabel('$\mathcal{L} (x)$', 'Interpreter','latex');
axis tight
aux2.Children(5).FaceColor = [1 0 0];
aux2.Children(4).FaceColor = [0 0 1];
aux2.Children(3).FaceColor = [0.5 0.5 0.5];
aux2.Children(2).FaceColor = [1 0.5 0];
aux2.Children(1).FaceColor = [0 0.5 0.5];
aux2.Children(5).DisplayName = "EKF";
aux2.Children(4).DisplayName = "UKF";
aux2.Children(3).DisplayName = "CKF";
aux2.Children(2).DisplayName = strcat("PF"," (N=",sprintf('%.0f)',Results{1}.N_points(4*n_config)));
aux2.Children(1).DisplayName = strcat("EnKF"," (N=",sprintf('%.0f)',Results{1}.N_points(5*n_config)));
aux2.OuterPosition(1) = 0;
aux2.OuterPosition(2) = 0.5;
aux2.OuterPosition(3) = 1;
aux2.OuterPosition(4) = 0.5;
if flag_ode == n_case
    legend("Location", "best","NumColumns",2);
end

T_comp = log10([Results{1}.Esf_mean([1,n_config+1, 2*n_config+1,...
    4*n_config, 5*n_config]); Results{2}.Esf_mean([1,n_config+1, 2*n_config+1,...
    4*n_config, 5*n_config]); Results{3}.Esf_mean([1,n_config+1, 2*n_config+1,...
    4*n_config, 5*n_config])]);
aux2 = subplot(2,1,2);
bar(1:3,T_comp);
xlabel("Simulation scenario")
ylabel('$\log_{10}(T_{comp})$ (s)', 'Interpreter','latex')
axis tight
aux2.Children(5).FaceColor = [1 0 0];
aux2.Children(4).FaceColor = [0 0 1];
aux2.Children(3).FaceColor = [0.5 0.5 0.5];
aux2.Children(2).FaceColor = [1 0.5 0];
aux2.Children(1).FaceColor = [0 0.5 0.5];
aux2.Children(5).DisplayName = "EKF";
aux2.Children(4).DisplayName = "UKF";
aux2.Children(3).DisplayName = "CKF";
aux2.Children(2).DisplayName = strcat("PF"," (N=",sprintf('%.0f)',Results{1}.N_points(4*n_config)));
aux2.Children(1).DisplayName = strcat("EnKF"," (N=",sprintf('%.0f)',Results{1}.N_points(5*n_config)));
aux2.OuterPosition(1) = 0;
aux2.OuterPosition(2) = 0;
aux2.OuterPosition(3) = 1;
aux2.OuterPosition(4) = 0.5;
aux.PaperUnits = 'centimeters';
aux.PaperPositionMode='manual';
aux.PaperSize = [14.8 13];
aux.PaperPosition = [0 0 14.8 13];
aux.PaperPositionMode='manual';
print(strcat(dir_save,"Unconst_Filt"),"-djpeg","-r300", '-image');

for flag_ode = 1:n_case
    nest = size(Results{flag_ode}.X_sim,1);

    aux = figure;
    for j = 1:nest
        aux2 = subplot(ceil(nest/2),2,j);
        plot(Plant{flag_ode}.T, Plant{flag_ode}.X(j,:),'-k');
        aux2.Children(1).DisplayName = "Plant";
        hold on
        if j == 2
            plot(Plant{flag_ode}.T, Plant{flag_ode}.Z(1,:),'xk');
            aux2.Children(1).DisplayName = "Measurement";
            legend("Location", "best","NumColumns",1);
        end
        if j == 3
            plot(Plant{flag_ode}.T, Plant{flag_ode}.Z(2,:),'xk');
            aux2.Children(1).DisplayName = "Measurement";
        end

        plot(Plant{flag_ode}.T, Results{flag_ode}.X_mean(j,:,1),'-',...
            Plant{flag_ode}.T, Results{flag_ode}.X_mean(j,:,n_config+2),'-',...
            Plant{flag_ode}.T, Results{flag_ode}.X_mean(j,:,2*n_config+3),'-',...
            Plant{flag_ode}.T, Results{flag_ode}.X_mean(j,:,3*n_config+4),'-',...
            Plant{flag_ode}.T, Results{flag_ode}.X_mean(j,:,4*n_config+5),'-');
        aux2.Children(5).Color = [1 0 0];
        aux2.Children(4).Color = [0 0 1];
        aux2.Children(3).Color = [0.5 0.5 0.5];
        aux2.Children(2).Color = [1 0.5 0];
        aux2.Children(1).Color = [0 0.5 0.5];
        aux2.Children(5).DisplayName = "EKF";
        aux2.Children(4).DisplayName = "UKF";
        aux2.Children(3).DisplayName = "CKF";
        aux2.Children(2).DisplayName = strcat("PF"," (N=",sprintf('%.0f)',Results{flag_ode}.N_points(4*n_config)));
        aux2.Children(1).DisplayName = strcat("EnKF"," (N=",sprintf('%.0f)',Results{flag_ode}.N_points(5*n_config)));

        aux2.OuterPosition(1) = (1-mod(j,2))/2;
        aux2.OuterPosition(2) = 1-(floor((j+1)/2))/ceil(nest/2);
        aux2.OuterPosition(3) = 0.5;
        aux2.OuterPosition(4) = 1/ceil(nest/2);
        xlabel("Time [h]")
        ylabel(sprintf("x_%.0f",j));
        axis tight

    end
    aux.PaperUnits = 'centimeters';
    aux.PaperPositionMode='manual';
    aux.PaperSize = [14.8 13];
    aux.PaperPosition = [0 0 14.8 13];
    aux.PaperPositionMode='manual';
    print(strcat(dir_save,sprintf("Unconst_Case%.0f",flag_ode)),"-djpeg","-r300", '-image');
end






















% Plotting the best simulations results for each state estimator
aux = figure;
% Computational time
T_best = log10([Results{1}.Esf_mean([Best_pos(1,1),n_config+Best_pos(1,2), 2*n_config+Best_pos(1,3),...
    3*n_config+Best_pos(1,4), 4*n_config+Best_pos(1,5)]); Results{2}.Esf_mean([Best_pos(2,1),n_config+Best_pos(2,2), 2*n_config+Best_pos(2,3),...
    3*n_config+Best_pos(2,4), 4*n_config+Best_pos(2,5)]); Results{3}.Esf_mean([Best_pos(3,1),n_config+Best_pos(3,2), 2*n_config+Best_pos(3,3),...
    3*n_config+Best_pos(3,4), 4*n_config+Best_pos(3,5)])]);
aux2 = subplot(3,1,2);
bar(1:3,T_best);
xlabel("Simulation scenario");
ylabel('$\log_{10}(T_{comp})$ (s)', 'Interpreter','latex');
axis tight
aux2.Children(5).FaceColor = [1 0 0];
aux2.Children(4).FaceColor = [0 0 1];
aux2.Children(3).FaceColor = [0.5 0.5 0.5];
aux2.Children(2).FaceColor = [1 0.5 0];
aux2.Children(1).FaceColor = [0 0.5 0.5];
aux2.Children(5).DisplayName = "EKF&S";
aux2.Children(4).DisplayName = "UKF&S";
aux2.Children(3).DisplayName = "CKF&S";
aux2.Children(2).DisplayName = "PF";
aux2.Children(1).DisplayName = "EnKF";
aux2.OuterPosition(1) = 0;
aux2.OuterPosition(2) = 1/3;
aux2.OuterPosition(3) = 1;
aux2.OuterPosition(4) = 1/3;

% Smoothing horizon
N_best = [Results{flag_ode}.N_smo(Best_pos(1,1:3)), Results{1}.N_points([3*n_config+Best_pos(1,4), 4*n_config+Best_pos(1,5)]);...
    Results{flag_ode}.N_smo(Best_pos(2,1:3)), Results{2}.N_points([3*n_config+Best_pos(2,4), 4*n_config+Best_pos(2,5)]);...
    Results{flag_ode}.N_smo(Best_pos(3,1:3)), Results{3}.N_points([3*n_config+Best_pos(3,4), 4*n_config+Best_pos(3,5)])];
aux2 = subplot(3,1,3);
bar(1:3,N_best(1:3,1:3));
axis tight
xlabel("Simulation scenario");
ylabel("Smoothing horizon");
aux2.Children(3).FaceColor = [1 0 0];
aux2.Children(2).FaceColor = [0 0 1];
aux2.Children(1).FaceColor = [0.5 0.5 0.5];
aux2.Children(3).DisplayName = "EKF&S";
aux2.Children(2).DisplayName = "UKF&S";
aux2.Children(1).DisplayName = "CKF&S";
aux2.OuterPosition(1) = 0;
aux2.OuterPosition(2) = 0;
aux2.OuterPosition(3) = 0.5;
aux2.OuterPosition(4) = 1/3;

% Number of particles
aux2 = subplot(3,1,1);
bar(1:3,N_best(1:3,4:5));
axis tight
xlabel("Simulation scenario");
ylabel("Number of particles");
aux2.Children(2).FaceColor = [1 0.5 0];
aux2.Children(1).FaceColor = [0 0.5 0.5];
aux2.Children(2).DisplayName = "PF";
aux2.Children(1).DisplayName = "EnKF";
aux2.OuterPosition(1) = 0.5;
aux2.OuterPosition(2) = 0;
aux2.OuterPosition(3) = 0.5;
aux2.OuterPosition(4) = 1/3;
aux2.XTickLabelRotation = 0;

% -Log-likelihood
aux2 = subplot(3,1,1);
bar(1:3,Best);
xlabel("Simulation scenario");
ylabel('$\mathcal{L} (x)$', 'Interpreter','latex');
axis tight
aux2.Children(5).FaceColor = [1 0 0];
aux2.Children(4).FaceColor = [0 0 1];
aux2.Children(3).FaceColor = [0.5 0.5 0.5];
aux2.Children(2).FaceColor = [1 0.5 0];
aux2.Children(1).FaceColor = [0 0.5 0.5];
aux2.Children(5).DisplayName = "EKF&S";
aux2.Children(4).DisplayName = "UKF&S";
aux2.Children(3).DisplayName = "CKF&S";
aux2.Children(2).DisplayName = "PF";
aux2.Children(1).DisplayName = "EnKF";
aux2.OuterPosition(1) = 0;
aux2.OuterPosition(2) = 2/3;
aux2.OuterPosition(3) = 1;
aux2.OuterPosition(4) = 1/3;
legend("Location", "best","NumColumns",2);
left = aux.Children(2).Position(1);
width = aux.Children(2).Position(3)+ aux.Children(2).Position(1);
for i = 3:numel(aux.Children)
    left = min(left, aux.Children(i).Position(1));
    width = max(aux.Children(i).Position(1)+aux.Children(i).Position(3),width);
end
for i = 2:numel(aux.Children)
    if aux.Children(i).Position(1) < 0.5
        aux.Children(i).Position(1) = left;
        if aux.Children(i).Position(3) > 0.5
            aux.Children(i).Position(3) = width-left;
        else
            aux.Children(i).Position(3) = 0.5-(1-width)-left;
        end
    end
end
aux.PaperUnits = 'centimeters';
aux.PaperPositionMode='manual';
aux.PaperSize = [14.8 13];
aux.PaperPosition = [0 0 14.8 13];
aux.PaperPositionMode='manual';
print(strcat(dir_save,"Unconst_Best"),"-djpeg","-r300", '-image');



for flag_ode = 1:n_case
    nest = size(Results{flag_ode}.X_sim,1);

    aux = figure;
    for j = 1:nest
        aux2 = subplot(ceil(nest/2),2,j);
        plot(Plant{flag_ode}.T, Plant{flag_ode}.X(j,:),'-k');
        aux2.Children(1).DisplayName = "Plant";
        hold on
        if j == 2
            plot(Plant{flag_ode}.T, Plant{flag_ode}.Z(1,:),'xk');
            aux2.Children(1).DisplayName = "Measurement";
            legend("Location", "best","NumColumns",1);
        end
        if j == 3
            plot(Plant{flag_ode}.T, Plant{flag_ode}.Z(2,:),'xk');
            aux2.Children(1).DisplayName = "Measurement";
        end

        plot(Plant{flag_ode}.T, Results{flag_ode}.X_mean(j,:,Best_pos(flag_ode,j)),'-',...
            Plant{flag_ode}.T, Results{flag_ode}.X_mean(j,:,n_config+Best_pos(flag_ode,j)),'-',...
            Plant{flag_ode}.T, Results{flag_ode}.X_mean(j,:,2*n_config+Best_pos(flag_ode,j)),'-',...
            Plant{flag_ode}.T, Results{flag_ode}.X_mean(j,:,3*n_config+Best_pos(flag_ode,j)),'-',...
            Plant{flag_ode}.T, Results{flag_ode}.X_mean(j,:,4*n_config+Best_pos(flag_ode,j)),'-');
        aux2.Children(5).Color = [1 0 0];
        aux2.Children(4).Color = [0 0 1];
        aux2.Children(3).Color = [0.5 0.5 0.5];
        aux2.Children(2).Color = [1 0.5 0];
        aux2.Children(1).Color = [0 0.5 0.5];
        if Best_pos(flag_ode,1) > 1
            aux2.Children(5).DisplayName = strcat("EKF&S"," (N=",sprintf('%.0f)',Results{flag_ode}.N_smo(Best_pos(flag_ode,1))));
        else
            aux2.Children(5).DisplayName = "EKF";
        end
        if Best_pos(flag_ode,2) > 1
            aux2.Children(4).DisplayName = strcat("UKF&S"," (N=",sprintf('%.0f)',Results{flag_ode}.N_smo(Best_pos(flag_ode,2))));
        else
            aux2.Children(4).DisplayName = "UKF";
        end
        if Best_pos(flag_ode,3) > 1
            aux2.Children(3).DisplayName = strcat("CKF&S"," (N=",sprintf('%.0f)',Results{flag_ode}.N_smo(Best_pos(flag_ode,3))));
        else
            aux2.Children(3).DisplayName = "CKF";
        end
        
        for i = 1:n_config
            aux2.XTickLabel{i} = Results{flag_ode}.N_smo(i);
        end
        aux2.XTickLabelRotation = 0;


        aux2.Children(2).DisplayName = strcat("PF"," (N=",sprintf('%.0f)',Results{flag_ode}.N_points(3*n_config+Best_pos(flag_ode,4))));
        aux2.Children(1).DisplayName = strcat("EnKF"," (N=",sprintf('%.0f)',Results{flag_ode}.N_points(4*n_config+Best_pos(flag_ode,5))));

        aux2.OuterPosition(1) = (1-mod(j,2))/2;
        aux2.OuterPosition(2) = 1-(floor((j+1)/2))/ceil(nest/2);
        aux2.OuterPosition(3) = 0.5;
        aux2.OuterPosition(4) = 1/ceil(nest/2);
        xlabel("Time [h]")
        ylabel(sprintf("x_%.0f",j));
        axis tight

    end
    aux.PaperUnits = 'centimeters';
    aux.PaperPositionMode='manual';
    aux.PaperSize = [14.8 13];
    aux.PaperPosition = [0 0 14.8 13];
    aux.PaperPositionMode='manual';
    print(strcat(dir_save,sprintf("Unconst_Case%.0f_Best",flag_ode)),"-djpeg","-r300", '-image');
end

end

