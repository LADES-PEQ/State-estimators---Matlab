function [Results, Table] = Plotting_Est(Plant, Results, n_sim_scen, N_smo, N_points, dir_save)
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
n_scen = (n_sim-1)/n_sim_scen;
n_config = 8; % Number of observer configurations evaluated for each simulated scenario
n_case = numel(Results); % Number of case studies

for flag_ode = 1:n_case
    [~,N_k,n_sim] = size(Results{flag_ode}.X_sim);
    ny = size(Results{flag_ode}.Y_sim,1);
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
        LogLikelihood = zeros(1,n_sim_scen);
        inv_P0 = inv(Results{flag_ode}.Pxx_sim(:,:,1,(m-1)*n_sim_scen+2));

        LogLikelihood(1,1) = LogLikelihood(1,1)+(Plant{flag_ode}.Z(1:ny,1)-...
                Results{flag_ode}.Y_sim(1:ny,1,(m-1)*n_sim_scen+1+1))'*...
                inv_R*(Plant{flag_ode}.Z(1:ny,1)-...
                Results{flag_ode}.Y_sim(1:ny,1,(m-1)*n_sim_scen+1+1));
        LogLikelihood(1,1) = LogLikelihood(1,1)+(Plant{flag_ode}.X(1:nest,1)-...
                        Results{flag_ode}.X_sim(1:nest,1,(m-1)*n_sim_scen+1+1))'*...
                        inv_P0*(Plant{flag_ode}.X(1:nest,1)-Results{flag_ode}.X_sim(1:nest,1,(m-1)*n_sim_scen+1+1));           

        LogLikelihood(1,:) = LogLikelihood(1,1);
        for k = 1:N_k
            if k>1
                for j = 1:n_sim_scen
                    LogLikelihood(1,j) = LogLikelihood(1,j)+(Plant{flag_ode}.Z(1:ny,k)-...
                        Results{flag_ode}.Y_sim(1:ny,k,(m-1)*n_sim_scen+1+j))'*...
                        inv_R*(Plant{flag_ode}.Z(1:ny,k)-...
                        Results{flag_ode}.Y_sim(1:ny,k,(m-1)*n_sim_scen+1+j));
                    LogLikelihood(1,j) = LogLikelihood(1,j)+(Plant{flag_ode}.X(1:nest,k)-...
                        Results{flag_ode}.X_sim(1:nest,k,(m-1)*n_sim_scen+1+j))'*...
                        inv_Q*(Plant{flag_ode}.X(1:nest,k)-Results{flag_ode}.X_sim(1:nest,k,(m-1)*n_sim_scen+1+j));
                end
            end

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
        Results{flag_ode}.LogLikelihood(1,m) = mean(LogLikelihood(1,1:n_sim_scen));
    end
end
Table = [];
for i = 1:3
    aux = Results{i}.LogLikelihood;
    aux2 = Results{i}.Esf_mean;

    for j = 1:8
        Table = [Table; [aux(j)/aux(1), aux2(j)/aux2(1), aux(40+j)/aux(1), aux2(40+j)/aux2(1), aux(8+j)/aux(1), aux2(8+j)/aux2(1), aux(16+j)/aux(1),...
            aux2(16+j)/aux2(1), aux(24+j)/aux(1), aux2(24+j)/aux2(1), aux(32+j)/aux(1), aux2(32+j)/aux2(1)]];
    end
end

%% Case study 2 (Paper figure)
% Simulation Scenario 2
Tagy = ["$C_s \left[\mathrm{kg/m}^3\right]$", "$C_x \left[\mathrm{kg/m}^3\right]$",...
    "$C_e \left[\mathrm{kg/m}^3\right]$", "$C_p \left[\mathrm{kg/m}^3\right]$"];

codcolor = [0,0,0;0,130,200;128,128,128;245,130,48;0,0,128;128,0,0;255,225,25;220,190,255;60,180,75]/255;
pos_y = [1,4];
aux = figure;
aux.PaperUnits = 'centimeters';
aux.PaperSize = [13.7 8];
aux.PaperPosition = [0 0 13.7 8];
ti_max = 1e5*ones(1,4);
for j = 1:4
    aux2 = subplot(2,2,j);
    plot(Plant{1}.T, Plant{1}.X(j,:), '-', Plant{1}.T, Results{1}.X_mean(j,:,1), '--',...
        Plant{1}.T, Results{1}.X_mean(j,:,44), ':', Plant{1}.T, Results{1}.X_mean(j,:,12), '-.',...
        Plant{1}.T, Results{1}.X_mean(j,:,40)), '--';

    xlabel('Time (h)', 'Interpreter','latex')
    ylabel(convertStringsToChars(Tagy(j)), 'Interpreter','latex');
    axis tight

    aux2.Children(1).LineStyle = '--';
    aux2.Children(1).Marker = '^';
    aux2.Children(1).MarkerSize = 3;
    aux2.Children(5).LineWidth = 1.5;
    aux2.Children(4).LineWidth = 1.5;
    aux2.Children(3).LineWidth = 1.5;
    aux2.Children(2).LineWidth = 1.5;
    aux2.Children(1).LineWidth = 1.5;
    aux2.Children(1).Color = codcolor(5,:);
    aux2.Children(2).Color = codcolor(4,:);
    aux2.Children(3).Color = codcolor(3,:);
    aux2.Children(4).Color = codcolor(2,:);
    aux2.Children(5).Color = codcolor(1,:);

    aux2.Children(5).DisplayName = "True";
    aux2.Children(4).DisplayName = "CEKF";
    aux2.Children(3).DisplayName = 'CSEKF $\left(N^S = N^*_4\right)$';
    aux2.Children(2).DisplayName = 'CUKFS $\left(N^S = N^*_4\right)$';
    aux2.Children(1).DisplayName = 'CEnKF $\left(N^E = N^*_8\right)$';
    
    if j == 1
        hold on
        plot(Plant{1}.T, Plant{1}.Z(1,:), 'X');
        aux2.Children(1).LineWidth = 0.5;
        aux2.Children(1).Color = codcolor(1,:);
        aux2.Children(1).DisplayName = "Measurement";
    end
    if j == 4
        hold on
        plot(aux2,Plant{1}.T, Plant{1}.Z(2,:), 'X');
        aux2.Children(1).LineWidth = 0.5;
        aux2.Children(1).Color = codcolor(1,:);
        aux2.Children(1).DisplayName = "Measurement";
    end        

    aux2.PositionConstraint = "outerposition";
    switch j
        case 1
            aux2.OuterPosition(1) = 0;
            aux2.OuterPosition(2) = 0.5;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 0.5;
        case 2
            aux2.OuterPosition(1) = 0.5;
            aux2.OuterPosition(2) = 0.5;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 0.5;
        case 3
            aux2.OuterPosition(1) = 0;
            aux2.OuterPosition(2) = 0;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 0.5;
        case 4
            aux2.OuterPosition(1) = 0.5;
            aux2.OuterPosition(2) = 0;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 0.5;
    end
end
ti_max = zeros(1,2);
Margin = 0.01;
for j = 1:4
        aux2 = aux.Children(4+1-j);
    switch j
        case 1
            ti_max(1) = max([aux.Children(4+1-j).Position(1),aux.Children(4+1-j-2).Position(1)]);
            ti_max(2) = max([aux.Children(4+1-j).Position(2),aux.Children(4+1-j-1).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin;
            aux2.Position(4) = 1-ti_max(2)-Margin-0.01;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2)+0.01;
        case 2
            ti_max(1) = max([aux.Children(4+1-j).Position(1),aux.Children(4+1-j-2).Position(1)]);
            ti_max(2) = max([aux.Children(4+1-j).Position(2),aux.Children(4+1-j+1).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 1-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
        case 3
            ti_max(1) = max([aux.Children(4+1-j).Position(1),aux.Children(4+1-j+2).Position(1)]);
            ti_max(2) = max([aux.Children(4+1-j).Position(2),aux.Children(4+1-j-1).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin;
            aux2.Position(4) = 0.5-ti_max(2)-Margin-0.01;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2)+0.01;

            aux2.NextPlot = 'add';
            plot(aux2, Plant{1}.T, Plant{1}.Z(1,:), 'X');
            aux2.Children(1).LineWidth = 0.5;
            aux2.Children(1).Color = codcolor(1,:);
            aux2.Children(1).DisplayName = "Measurement";
           
            legend(aux2.Children([1,6:-1:2]),'Interpreter','latex')
            aux2.Legend.NumColumns = 1;
            aux2.Legend.FontSize = 6.5;
            aux2.Legend.Location = 'Northwest';
            aux2.Legend.AutoUpdate = 'off';
            pause(0.01)
            delete(aux2.Children(1));          
            
        case 4
            ti_max(1) = max([aux.Children(4+1-j).Position(1),aux.Children(4+2-j+2).Position(1)]);
            ti_max(2) = max([aux.Children(4+1-j).Position(2),aux.Children(4+2-j+1).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 0.5-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
    end
end

%13.70499cm
axis tight
print(strcat(dir_save,"Fig1_CS2"),"-depsc2","-r1000","-vector");

%% 
% Simulation Scenario 3
codcolor = [0,0,0;0,130,200;128,128,128;245,130,48;0,0,128;128,0,0;255,225,25;220,190,255;60,180,75]/255;
pos_y = [1,4];
aux = figure;
aux.PaperUnits = 'centimeters';
aux.PaperSize = [13.7 8];
aux.PaperPosition = [0 0 13.7 8];
ti_max = 1e5*ones(1,4);
for j = 1:4
    aux2 = subplot(2,2,j);
    plot(Plant{2}.T, Plant{2}.X(j,:), '-', Plant{2}.T, Results{2}.X_mean(j,:,1), '--',...
        Plant{2}.T, Results{2}.X_mean(j,:,3), ':', Plant{2}.T, Results{2}.X_mean(j,:,43), '-.',...
        Plant{2}.T, Results{2}.X_mean(j,:,9), '--', Plant{2}.T, Results{2}.X_mean(j,:,10)), '--';

    xlabel('Time (h)', 'Interpreter','latex')
    ylabel(convertStringsToChars(Tagy(j)), 'Interpreter','latex');
    axis tight
    aux2.Children(1).LineStyle = '--';
    aux2.Children(1).Marker = '^';
    aux2.Children(1).MarkerSize = 3;
    aux2.Children(2).LineStyle = '--';
    aux2.Children(2).Marker = 'square';
    aux2.Children(2).MarkerSize = 3;
    aux2.Children(6).LineWidth = 1.5;
    aux2.Children(5).LineWidth = 1.5;
    aux2.Children(4).LineWidth = 1.5;
    aux2.Children(3).LineWidth = 1.5;
    aux2.Children(2).LineWidth = 1.5;
    aux2.Children(1).LineWidth = 1.5;
    aux2.Children(1).Color = codcolor(6,:);
    aux2.Children(2).Color = codcolor(5,:);
    aux2.Children(3).Color = codcolor(4,:);
    aux2.Children(4).Color = codcolor(3,:);
    aux2.Children(5).Color = codcolor(2,:);
    aux2.Children(6).Color = codcolor(1,:);

    aux2.Children(6).DisplayName = "True";
    aux2.Children(5).DisplayName = "CEKF";
    aux2.Children(4).DisplayName = 'CEKFS $\left(N^S = N^*_3\right)$';
    aux2.Children(3).DisplayName = 'CSEKF $\left(N^S = N^*_3\right)$';
    aux2.Children(2).DisplayName = 'CUKF';
    aux2.Children(1).DisplayName = 'CUKFS $\left(N^S = N^*_2\right)$';
    
    if j == 1
        hold on
        plot(Plant{1}.T, Plant{1}.Z(1,:), 'X');
        aux2.Children(1).LineWidth = 0.5;
        aux2.Children(1).Color = codcolor(1,:);
        aux2.Children(1).DisplayName = "Measurement";
    end
    if j == 4
        hold on
        plot(aux2,Plant{1}.T, Plant{1}.Z(2,:), 'X');
        aux2.Children(1).LineWidth = 0.5;
        aux2.Children(1).Color = codcolor(1,:);
        aux2.Children(1).DisplayName = "Measurement";
    end        

    switch j
        case 1
            aux2.OuterPosition(1) = 0;
            aux2.OuterPosition(2) = 0.5;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 0.5;
        case 2
            aux2.OuterPosition(1) = 0.5;
            aux2.OuterPosition(2) = 0.5;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 0.5;
        case 3
            aux2.OuterPosition(1) = 0;
            aux2.OuterPosition(2) = 0;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 0.5;
        case 4
            aux2.OuterPosition(1) = 0.5;
            aux2.OuterPosition(2) = 0;
            aux2.OuterPosition(3) = 0.5;
            aux2.OuterPosition(4) = 0.5;
    end
end
ti_max = zeros(1,2);
Margin = 0.01;
for j = 1:4
        aux2 = aux.Children(4+1-j);
    switch j
        case 1
            ti_max(1) = max([aux.Children(4+1-j).Position(1),aux.Children(4+1-j-2).Position(1)]);
            ti_max(2) = max([aux.Children(4+1-j).Position(2),aux.Children(4+1-j-1).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin-0.01;
            aux2.Position(4) = 1-ti_max(2)-Margin-0.01;
            aux2.Position(1) = ti_max(1)+0.01;
            aux2.Position(2) = ti_max(2)+0.01;
        case 2
            ti_max(1) = max([aux.Children(4+1-j).Position(1),aux.Children(4+1-j-2).Position(1)]);
            ti_max(2) = max([aux.Children(4+1-j).Position(2),aux.Children(4+1-j+1).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 1-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
        case 3
            ti_max(1) = max([aux.Children(4+1-j).Position(1),aux.Children(4+1-j+2).Position(1)]);
            ti_max(2) = max([aux.Children(4+1-j).Position(2),aux.Children(4+1-j-1).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin;
            aux2.Position(4) = 0.5-ti_max(2)-Margin-0.013;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2)+0.013;

            aux2.NextPlot = 'add';
            plot(aux2, Plant{2}.T, Plant{2}.Z(1,:), 'X');
            aux2.Children(1).LineWidth = 0.5;
            aux2.Children(1).Color = codcolor(1,:);
            aux2.Children(1).DisplayName = "Measurement";
            
            legend(aux2.Children([1,7:-1:2]),'Interpreter','latex')
            aux2.Legend.NumColumns = 1;
            aux2.Legend.FontSize = 5.9;
            aux2.Legend.Location = 'best';
            aux2.Legend.AutoUpdate = 'off';
            pause(0.01)
            delete(aux2.Children(1));          
            
        case 4
            ti_max(1) = max([aux.Children(4+1-j).Position(1),aux.Children(4+2-j+2).Position(1)]);
            ti_max(2) = max([aux.Children(4+1-j).Position(2),aux.Children(4+2-j+1).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 0.5-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
            
    end
end
axis tight
print(strcat(dir_save,"Fig2_CS2"),"-depsc2","-r1000","-vector");
%% Case study 1 (Extra figures)
% Calculating relative outputs
Tagy = ["$\overline{J}$", "$\overline{t}_{c}$"];
% Plotting the simulations results under forward-backward filtering
aux = figure;
aux.PaperUnits = 'centimeters';
aux.PaperPositionMode='manual';
aux.PaperSize = [14.8 13];
aux.PaperPosition = [0 0 14.8 13];
for i = 1:n_case
    for j = 1:2

        aux2 = subplot(3,2,(i-1)*2+j);
        bar(0:n_config-1,Table((i-1)*8+1:i*8,[1,3,5,7]+(j-1)));
        xlabel('Smoothing horizon', 'Interpreter','latex')
        ylabel(convertStringsToChars(Tagy(j)), 'Interpreter','latex');
        axis tight
        aux2.Children(4).FaceColor = codcolor(2,:);
        aux2.Children(3).FaceColor = codcolor(3,:);
        aux2.Children(2).FaceColor = codcolor(4,:);
        aux2.Children(1).FaceColor = codcolor(5,:);
        aux2.Children(4).DisplayName = "CEKFS";
        aux2.Children(3).DisplayName = "CSEKFS";
        aux2.Children(2).DisplayName = "CUKFS";
        aux2.Children(1).DisplayName = "CCKFS";
        for m = 1:n_config
            aux2.XTickLabel{m} = Results{i}.N_smo(m);
        end

        switch (i-1)*2+j
            case 1
                aux2.OuterPosition(1) = 0;
                aux2.OuterPosition(2) = 2/3;
                aux2.OuterPosition(3) = 0.5;
                aux2.OuterPosition(4) = 1/3;
            case 2
                aux2.OuterPosition(1) = 0.5;
                aux2.OuterPosition(2) = 2/3;
                aux2.OuterPosition(3) = 0.5;
                aux2.OuterPosition(4) = 1/3;
            case 3
                aux2.OuterPosition(1) = 0;
                aux2.OuterPosition(2) = 1/3;
                aux2.OuterPosition(3) = 0.5;
                aux2.OuterPosition(4) = 1/3;
            case 4
                aux2.OuterPosition(1) = 0.5;
                aux2.OuterPosition(2) = 1/3;
                aux2.OuterPosition(3) = 0.5;
                aux2.OuterPosition(4) = 1/3;
            case 5
                aux2.OuterPosition(1) = 0;
                aux2.OuterPosition(2) = 0;
                aux2.OuterPosition(3) = 0.5;
                aux2.OuterPosition(4) = 1/3;
            case 6
                aux2.OuterPosition(1) = 0.5;
                aux2.OuterPosition(2) = 0;
                aux2.OuterPosition(3) = 0.5;
                aux2.OuterPosition(4) = 1/3;
        end
    end
end
ti_max = zeros(1,2);
Margin = 0.01;
for j = 1:6
        aux2 = aux.Children(6+1-j);
    switch j
        case 1
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+1-j-2).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+1-j-1).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin;
            aux2.Position(4) = 1-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
        case 2
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+1-j-2).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+1-j+1).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 1-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
        case 3            
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+1-j+2).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+1-j-1).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin;
            aux2.Position(4) = 2/3-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
            
        case 4
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+1-j+2).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+1-j+1).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 2/3-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);

        case 5       
            ti_max(1) = max([aux.Children(6+1-j+2).Position(1),aux.Children(6+1-j+4).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin;
            aux2.Position(4) = 1/3-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);

       case 6           
            legend(aux2.Children([4:-1:1]),'Interpreter','latex')
            aux2.Legend.NumColumns = 1;
            aux2.Legend.FontSize = 6.5;
            aux2.Legend.Location = 'northwest';
            aux2.Legend.AutoUpdate = 'off';

            ti_max(1) = max([aux.Children(6+2-j).Position(1),aux.Children(6+2-j+2).Position(1)]);
            ti_max(2) = max([aux.Children(6+2-j).Position(2),aux.Children(6+2-j+1).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 1/3-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
    end
end
print(strcat(dir_save,"Const_Filt_Smooth"),"-djpeg","-r300", '-image');

%%

% Plotting the simulations results subject to PF or EnKF estimations
aux = figure;
aux.PaperUnits = 'centimeters';
aux.PaperPositionMode='manual';
aux.PaperSize = [14.8 13];
aux.PaperPosition = [0 0 14.8 13];
for i = 1:n_case
    for j = 1:2
            aux2 = subplot(3,2,(i-1)*2+j);
            bar(0:n_config-1,Table((i-1)*8+1:i*8,[9,11]+(j-1)));
            xlabel("Ensemble size", 'Interpreter','latex')
            ylabel('$\overline{J}$', 'Interpreter','latex');
            axis tight
            aux2.Children(2).FaceColor = codcolor(6,:);
            aux2.Children(1).FaceColor = codcolor(7,:);
            aux2.Children(2).DisplayName = "CPF";
            aux2.Children(1).DisplayName = "CEnKF";
            for m = 1:n_config
                aux2.XTickLabel{m} = Results{i}.N_points(n_config*3+m);
            end
            aux2.XTickLabelRotation = 0;
       switch (i-1)*2+j
           case 1
                aux2.OuterPosition(1) = 0;
                aux2.OuterPosition(2) = 2/3;
                aux2.OuterPosition(3) = 0.5;
                aux2.OuterPosition(4) = 1/3;
            case 2
                aux2.OuterPosition(1) = 0.5;
                aux2.OuterPosition(2) = 2/3;
                aux2.OuterPosition(3) = 0.5;
                aux2.OuterPosition(4) = 1/3;
            case 3
                aux2.OuterPosition(1) = 0;
                aux2.OuterPosition(2) = 1/3;
                aux2.OuterPosition(3) = 0.5;
                aux2.OuterPosition(4) = 1/3;
            case 4
                aux2.OuterPosition(1) = 0.5;
                aux2.OuterPosition(2) = 1/3;
                aux2.OuterPosition(3) = 0.5;
                aux2.OuterPosition(4) = 1/3;
            case 5
                aux2.OuterPosition(1) = 0;
                aux2.OuterPosition(2) = 0;
                aux2.OuterPosition(3) = 0.5;
                aux2.OuterPosition(4) = 1/3;
            case 6
                aux2.OuterPosition(1) = 0.5;
                aux2.OuterPosition(2) = 0;
                aux2.OuterPosition(3) = 0.5;
                aux2.OuterPosition(4) = 1/3;
       end
    end
end
ti_max = zeros(1,2);
Margin = 0.01;
for j = 1:6
        aux2 = aux.Children(6+1-j);
    switch j
        case 1
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+1-j-2).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+1-j-1).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin;
            aux2.Position(4) = 1-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
        case 2
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+1-j-2).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+1-j+1).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 1-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
        case 3            
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+1-j+2).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+1-j-1).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin;
            aux2.Position(4) = 2/3-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
            
        case 4
            ti_max(1) = max([aux.Children(6+1-j).Position(1),aux.Children(6+1-j+2).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2),aux.Children(6+1-j+1).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 2/3-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);

        case 5       
            ti_max(1) = max([aux.Children(6+1-j+2).Position(1),aux.Children(6+1-j+4).Position(1)]);
            ti_max(2) = max([aux.Children(6+1-j).Position(2)]);
            aux2.Position(3) = 0.5-ti_max(1)-Margin;
            aux2.Position(4) = 1/3-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);

       case 6           
            legend(aux2.Children([2,1]),'Interpreter','latex')
            aux2.Legend.NumColumns = 1;
            aux2.Legend.FontSize = 6.5;
            aux2.Legend.Location = 'northwest';
            aux2.Legend.AutoUpdate = 'off';

            ti_max(1) = max([aux.Children(6+2-j).Position(1),aux.Children(6+2-j+2).Position(1)]);
            ti_max(2) = max([aux.Children(6+2-j).Position(2),aux.Children(6+2-j+1).Position(2)]);
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 1/3-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
    end
end
print(strcat(dir_save,"Const_Filt_Part"),"-djpeg","-r300", '-image');
%%

% Plotting the best simulations results under forward filtering
aux = figure;
aux.PaperUnits = 'centimeters';
aux.PaperPositionMode='manual';
aux.PaperSize = [14.8 13];
aux.PaperPosition = [0 0 14.8 13];
for j = 1:2
aux2 = subplot(2,1,j);

bar(1:3,Table([1,n_config+1,2*n_config+1], [1,3,5,7,9,11]));
xlabel('Simulation scenario', 'Interpreter','latex')
ylabel(convertStringsToChars(Tagy(j)), 'Interpreter','latex');
axis tight
aux2.Children(6).FaceColor = codcolor(2,:);
aux2.Children(5).FaceColor = codcolor(3,:);
aux2.Children(4).FaceColor = codcolor(4,:);
aux2.Children(3).FaceColor = codcolor(5,:);
aux2.Children(2).FaceColor = codcolor(6,:);
aux2.Children(1).FaceColor = codcolor(7,:);
aux2.Children(6).DisplayName = "CEKF";
aux2.Children(5).DisplayName = "CSEKF";
aux2.Children(4).DisplayName = "CUKF";
aux2.Children(3).DisplayName = "CCKF";
aux2.Children(2).DisplayName = strcat("CPF"," (N=",sprintf('%.0f)',Results{1}.N_points(4*n_config)));
aux2.Children(1).DisplayName = strcat("CEnKF"," (N=",sprintf('%.0f)',Results{1}.N_points(5*n_config)));
switch j
    case 1
        aux2.OuterPosition(1) = 0;
        aux2.OuterPosition(2) = 0.5;
        aux2.OuterPosition(3) = 1;
        aux2.OuterPosition(4) = 0.5;
    case 2
        aux2.OuterPosition(1) = 0;
        aux2.OuterPosition(2) = 0;
        aux2.OuterPosition(3) = 1;
        aux2.OuterPosition(4) = 0.5;
end
end
ti_max = zeros(1,2);
Margin = 0.01;
for j = 1:2
    aux2 = aux.Children(2+1-j);
    switch j
        case 1
            legend(aux2.Children([6:-1:1]),'Interpreter','latex')
            aux2.Legend.NumColumns = 2;
            aux2.Legend.FontSize = 6.5;
            aux2.Legend.Location = 'northwest';
            aux2.Legend.AutoUpdate = 'off';

            ti_max(1) = max([aux.Children(2+2-j).Position(1),aux.Children(2+2-j).Position(1)]);
            ti_max(2) = max(aux.Children(2+2-j).Position(2));
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 1-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
        case 2
            ti_max(1) = max([aux.Children(1).Position(1),aux.Children(3).Position(1)]);
            ti_max(2) = max(aux.Children(1).Position(2));
            aux2.Position(3) = 1-ti_max(1)-Margin;
            aux2.Position(4) = 0.5-ti_max(2)-Margin;
            aux2.Position(1) = ti_max(1);
            aux2.Position(2) = ti_max(2);
    end
end
print(strcat(dir_save,"Const_Filt"),"-djpeg","-r300", '-image');

%%
% Definition of the best configuration simulated for each observer
BestJ = zeros(n_case, 6);
BestT = zeros(n_case, 6);
Best_pos = zeros(n_case, 6);

for i = 1:n_case
    for j = 1:6
        [BestJ(i,j), Best_pos(i,j)] = min(Table((i-1)*n_config+1:i*n_config,(j-1)*2+1));
    end
end

for i = 1:n_case
    for j = 1:6
        BestT(i,j) = Table((i-1)*n_config+Best_pos(i,j),(j-1)*2+2);
    end
end

BestN = Results{1}.N_smo(Best_pos);
BestN(:,5:6) = Results{1}.N_points(32+Best_pos(:,end-1:end));

% Plotting the best simulations results for each state estimator
aux = figure;
aux.PaperUnits = 'centimeters';
aux.PaperPositionMode='manual';
aux.PaperSize = [14.8 18.5];
aux.PaperPosition = [0 0 14.8 18.5];

% Computational time
aux2 = subplot(3,1,2);
bar(1:3,BestT);
xlabel('Simulation scenario', 'Interpreter','latex');
ylabel('$\overline{t}_{c}$', 'Interpreter','latex');
axis tight
aux2.Children(6).FaceColor = codcolor(2,:);
aux2.Children(5).FaceColor = codcolor(3,:);
aux2.Children(4).FaceColor = codcolor(4,:);
aux2.Children(3).FaceColor = codcolor(5,:);
aux2.Children(2).FaceColor = codcolor(6,:);
aux2.Children(1).FaceColor = codcolor(7,:);
aux2.Children(6).DisplayName = "CEKFS";
aux2.Children(5).DisplayName = "CSEKF";
aux2.Children(4).DisplayName = "CUKFS";
aux2.Children(3).DisplayName = "CCKFS";
aux2.Children(2).DisplayName = "CPF";
aux2.Children(1).DisplayName = "CEnKF";
aux2.OuterPosition(1) = 0;
aux2.OuterPosition(2) = 1/3;
aux2.OuterPosition(3) = 1;
aux2.OuterPosition(4) = 1/3;

% Smoothing horizon
aux2 = subplot(3,1,3);
bar(1:3,BestN(1:3,1:4));
axis tight
xlabel('Simulation scenario','Interpreter','latex');
ylabel('Smoothing horizon','Interpreter','latex');
aux2.Children(4).FaceColor = codcolor(2,:);
aux2.Children(3).FaceColor = codcolor(3,:);
aux2.Children(2).FaceColor = codcolor(4,:);
aux2.Children(1).FaceColor = codcolor(5,:);
aux2.Children(4).DisplayName = "CEKFS";
aux2.Children(3).DisplayName = "CSEKFS";
aux2.Children(2).DisplayName = "CUKFS";
aux2.Children(1).DisplayName = "CCKFS";
aux2.OuterPosition(1) = 0;
aux2.OuterPosition(2) = 0;
aux2.OuterPosition(3) = 0.5;
aux2.OuterPosition(4) = 1/3;

% Number of particles
aux2 = subplot(3,1,1);
bar(1:3,BestN(1:3,5:6));
axis tight
xlabel('Simulation scenario','Interpreter','latex');
ylabel('Ensemble size','Interpreter','latex');
aux2.Children(2).FaceColor = codcolor(6,:);
aux2.Children(1).FaceColor = codcolor(7,:);
aux2.Children(2).DisplayName = "CPF";
aux2.Children(1).DisplayName = "CEnKF";
aux2.OuterPosition(1) = 0.5;
aux2.OuterPosition(2) = 0;
aux2.OuterPosition(3) = 0.5;
aux2.OuterPosition(4) = 1/3;
aux2.XTickLabelRotation = 0;

% -Log-likelihood
aux2 = subplot(3,1,1);
bar(1:3,BestJ);
xlabel('Simulation scenario', 'Interpreter','latex');
ylabel('$\overline{J}$', 'Interpreter','latex');
axis tight
aux2.Children(6).FaceColor = codcolor(2,:);
aux2.Children(5).FaceColor = codcolor(3,:);
aux2.Children(4).FaceColor = codcolor(4,:);
aux2.Children(3).FaceColor = codcolor(5,:);
aux2.Children(2).FaceColor = codcolor(6,:);
aux2.Children(1).FaceColor = codcolor(7,:);
aux2.Children(6).DisplayName = "CEKFS";
aux2.Children(5).DisplayName = "CSEKFS";
aux2.Children(4).DisplayName = "CUKFS";
aux2.Children(3).DisplayName = "CCKFS";
aux2.Children(2).DisplayName = "CPF";
aux2.Children(1).DisplayName = "CEnKF";
aux2.OuterPosition(1) = 0;
aux2.OuterPosition(2) = 2/3;
aux2.OuterPosition(3) = 1;
aux2.OuterPosition(4) = 1/3;


legend(aux2.Children([6:-1:1]),'Interpreter','latex')
aux2.Legend.NumColumns = 3;
aux2.Legend.FontSize = 6.5;
aux2.Legend.Location = 'north';

ti_max = zeros(1,2);
Margin = 0.01;
left = aux.Children(2).Position(1);
width = aux.Children(2).Position(3)+ aux.Children(2).Position(1);
for i = 3:numel(aux.Children)
    left = min(left, aux.Children(i).Position(1));
    width = max(aux.Children(i).Position(1)+aux.Children(i).Position(3),width);
end
for i = 2:numel(aux.Children)
    if aux.Children(i).Position(1) < 0.5
        aux.Children(i).Position(1) = left+0.02;
        if aux.Children(i).Position(3) > 0.5
            aux.Children(i).Position(3) = width-left;
        else
            aux.Children(i).Position(3) = 0.5-(1-width)-left;
        end
    else
        aux.Children(i).Position(3) = aux.Children(i).Position(3) + 0.02;
    end
end
print(strcat(dir_save,"Const_Best"),"-djpeg","-r300", '-image');

%%
Tagy = ["$C_s \left[\mathrm{kg/m}^3\right]$", "$C_x \left[\mathrm{kg/m}^3\right]$",...
    "$C_e \left[\mathrm{kg/m}^3\right]$", "$C_p \left[\mathrm{kg/m}^3\right]$",...
    "$\Delta D \left[\mathrm{kg/m}^3\right]$", "$\Delta C_{s0} \left[\mathrm{kg/m}^3\right]$"];

for i = 1:n_case
    nest = size(Results{i}.X_sim,1);

    aux = figure;
    aux.PaperUnits = 'centimeters';
    aux.PaperPositionMode='manual';
    if i<3
        aux.PaperSize = [14.8 13];
        aux.PaperPosition = [0 0 14.8 13];
    else
        aux.PaperSize = [14.8 18.5];
        aux.PaperPosition = [0 0 14.8 18.5];
    end
    for j = 1:nest
        aux2 = subplot(ceil(nest/2),2,j);
        if j == 1
            plot(Plant{i}.T, Plant{i}.Z(1,:),'xk');
            hold on
            aux2.Children(1).DisplayName = "Measurement";
        end
        if j == 4
            plot(Plant{i}.T, Plant{i}.Z(2,:),'xk');
            hold on
            aux2.Children(1).DisplayName = "Measurement";
        end

        plot(Plant{i}.T, Plant{i}.X(j,:),'-k');
        hold on
        aux2.Children(1).DisplayName = "Plant";
        aux2.Children(1).LineWidth = 1.5;

        plot(Plant{i}.T, Results{i}.X_mean(j,:,1),'-',...
            Plant{i}.T, Results{i}.X_mean(j,:,5*n_config+1),'-',...
            Plant{i}.T, Results{i}.X_mean(j,:,n_config+1),'-',...
            Plant{i}.T, Results{i}.X_mean(j,:,2*n_config+1),'-',...
            Plant{i}.T, Results{i}.X_mean(j,:,3*n_config+8),'-',...
            Plant{i}.T, Results{i}.X_mean(j,:,4*n_config+8),'-');
        aux2.Children(6).Color = codcolor(2,:);
        aux2.Children(5).Color = codcolor(3,:);
        aux2.Children(4).Color = codcolor(4,:);
        aux2.Children(3).Color = codcolor(5,:);
        aux2.Children(2).Color = codcolor(6,:);
        aux2.Children(1).Color = codcolor(7,:);

        aux2.Children(6).LineWidth = 1.5;
        aux2.Children(5).LineWidth = 1.5;
        aux2.Children(4).LineWidth = 1.5;
        aux2.Children(3).LineWidth = 1.5;
        aux2.Children(2).LineWidth = 1.5;
        aux2.Children(1).LineWidth = 1.5;

        aux2.Children(6).DisplayName = "CEKF";
        aux2.Children(5).DisplayName = "CSEKF";
        aux2.Children(4).DisplayName = "CUKF";
        aux2.Children(3).DisplayName = "CCKF";

        aux2.Children(2).DisplayName = strcat("CPF"," (N=",sprintf('%.0f)',5000));
        aux2.Children(1).DisplayName = strcat("CEnKF"," (N=",sprintf('%.0f)',5000));



        aux2.OuterPosition(1) = (1-mod(j,2))/2;
        aux2.OuterPosition(2) = 1-(floor((j+1)/2))/ceil(nest/2);
        aux2.OuterPosition(3) = 0.5;
        aux2.OuterPosition(4) = 1/ceil(nest/2);
        xlabel('Time [h]', 'Interpreter','latex')
        ylabel(convertStringsToChars(Tagy(j)), 'Interpreter','latex');
        axis tight
        if j == 6
            aux2.Children(1).Parent.YLim =[-0.1, 0.1];
        end
    end

    ti_max = zeros(1,2);
    Margin = 0.02;
    for j = 1:nest
        aux2 = aux.Children(nest+1-j);
        switch j
            case 1
                if i < 3
                    ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+1-j-2).Position(1)]);
                else
                    ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+1-j-2).Position(1),aux.Children(nest+1-j-4).Position(1)]);
                end
                ti_max(2) = max([aux.Children(nest+1-j).Position(2),aux.Children(nest+1-j-1).Position(2)]);
                aux2.Position(3) = aux2.OuterPosition(3)+aux2.OuterPosition(1)-ti_max(1)-Margin;
                aux2.Position(4) = aux2.OuterPosition(4)+aux2.OuterPosition(2)-ti_max(2)-Margin;
                aux2.Position(1) = ti_max(1);
                aux2.Position(2) = ti_max(2);

            case 2
                if i < 3
                    ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+1-j-2).Position(1)]);
                else
                    ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+1-j-2).Position(1),aux.Children(nest+1-j-4).Position(1)]);
                end
                ti_max(2) = max([aux.Children(nest+1-j).Position(2),aux.Children(nest+1-j+1).Position(2)]);
                aux2.Position(3) = aux2.OuterPosition(3)+aux2.OuterPosition(1)-ti_max(1)-Margin;
                aux2.Position(4) = aux2.OuterPosition(4)+aux2.OuterPosition(2)-ti_max(2)-Margin;
                aux2.Position(1) = ti_max(1);
                aux2.Position(2) = ti_max(2);
            case 3
                if i < 3
                    ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+1-j+2).Position(1)]);
                else
                    ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+1-j-2).Position(1),aux.Children(nest+1-j+2).Position(1)]);
                end
                ti_max(2) = max([aux.Children(nest+1-j).Position(2),aux.Children(nest+1-j-1).Position(2)]);

                aux2.Position(3) = aux2.OuterPosition(3)+aux2.OuterPosition(1)-ti_max(1)-Margin;
                aux2.Position(4) = aux2.OuterPosition(4)+aux2.OuterPosition(2)-ti_max(2)-Margin;
                aux2.Position(1) = ti_max(1);
                aux2.Position(2) = ti_max(2);


                aux2.NextPlot = 'add';
                plot(aux2, Plant{2}.T, Plant{2}.Z(1,:), 'X');
                aux2.Children(1).LineWidth = 0.5;
                aux2.Children(1).Color = codcolor(1,:);
                aux2.Children(1).DisplayName = "Measurement";
                legend(aux2.Children([1,8:-1:2]),'Interpreter','latex')
                aux2.Legend.NumColumns = 1;
                aux2.Legend.FontSize = 6.5;
                aux2.Legend.Location = 'northwest';
                aux2.Legend.AutoUpdate = 'off';
                pause(0.01)
                delete(aux2.Children(1));

            case 4
                ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+2-j+2).Position(1)]);
                ti_max(2) = max([aux.Children(nest+1-j).Position(2),aux.Children(nest+2-j+1).Position(2)]);
                aux2.Position(3) = aux2.OuterPosition(3)+aux2.OuterPosition(1)-ti_max(1)-Margin;
                aux2.Position(4) = aux2.OuterPosition(4)+aux2.OuterPosition(2)-ti_max(2)-Margin;
                aux2.Position(1) = ti_max(1);
                aux2.Position(2) = ti_max(2);

            case 5
                ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+2-j+2).Position(1)]);
                ti_max(2) = max([aux.Children(nest+1-j).Position(2),aux.Children(nest+1-j-1).Position(2)]);
                aux2.Position(3) = aux2.OuterPosition(3)+aux2.OuterPosition(1)-ti_max(1)-Margin;
                aux2.Position(4) = aux2.OuterPosition(4)+aux2.OuterPosition(2)-ti_max(2)-Margin;
                aux2.Position(1) = ti_max(1);
                aux2.Position(2) = ti_max(2);


            case 6

                ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+2-j+2).Position(1)]);
                ti_max(2) = max([aux.Children(nest+1-j).Position(2),aux.Children(nest+1-j+1).Position(2)]);
                aux2.Position(3) = aux2.OuterPosition(3)+aux2.OuterPosition(1)-ti_max(1)-Margin;
                aux2.Position(4) = aux2.OuterPosition(4)+aux2.OuterPosition(2)-ti_max(2)-Margin;
                aux2.Position(1) = ti_max(1);
                aux2.Position(2) = ti_max(2);
        end
%         if i == 3
%             aux2.Position(3) = aux2.Position(3)+0.03;
%             aux2.Position(1) = aux2.Position(1)-0.03;
%         end
    end
    print(strcat(dir_save,sprintf("Const_Case%.0f",i)),"-djpeg","-r300", '-image');
end


%%
Tagy = ["$C_s \left[\mathrm{kg/m}^3\right]$", "$C_x \left[\mathrm{kg/m}^3\right]$",...
    "$C_e \left[\mathrm{kg/m}^3\right]$", "$C_p \left[\mathrm{kg/m}^3\right]$",...
    "$\Delta D \left[\mathrm{kg/m}^3\right]$", "$\Delta C_{s0} \left[\mathrm{kg/m}^3\right]$"];

for i = 1:n_case
    nest = size(Results{i}.X_sim,1);

    aux = figure;
    aux.PaperUnits = 'centimeters';
    aux.PaperPositionMode='manual';
    if i<3
        aux.PaperSize = [14.8 13];
        aux.PaperPosition = [0 0 14.8 13];
    else
        aux.PaperSize = [14.8 18.5];
        aux.PaperPosition = [0 0 14.8 18.5];
    end
    for j = 1:nest
        aux2 = subplot(ceil(nest/2),2,j);
        if j == 1
            plot(Plant{i}.T, Plant{i}.Z(1,:),'xk');
            hold on
            aux2.Children(1).DisplayName = "Measurement";
        end
        if j == 4
            plot(Plant{i}.T, Plant{i}.Z(2,:),'xk');
            hold on
            aux2.Children(1).DisplayName = "Measurement";
        end

        plot(Plant{i}.T, Plant{i}.X(j,:),'-k');
        hold on
        aux2.Children(1).DisplayName = "Plant";
        aux2.Children(1).LineWidth = 1.5;

        plot(Plant{i}.T, Results{i}.X_mean(j,:,Best_pos(i,1)),'-',...
            Plant{i}.T, Results{i}.X_mean(j,:,5*n_config+Best_pos(i,2)),'-',...
            Plant{i}.T, Results{i}.X_mean(j,:,n_config+Best_pos(i,3)),'-',...
            Plant{i}.T, Results{i}.X_mean(j,:,2*n_config+Best_pos(i,4)),'-',...
            Plant{i}.T, Results{i}.X_mean(j,:,3*n_config+Best_pos(i,5)),'-',...
            Plant{i}.T, Results{i}.X_mean(j,:,4*n_config+Best_pos(i,6)),'-');
        aux2.Children(6).Color = codcolor(2,:);
        aux2.Children(5).Color = codcolor(3,:);
        aux2.Children(4).Color = codcolor(4,:);
        aux2.Children(3).Color = codcolor(5,:);
        aux2.Children(2).Color = codcolor(6,:);
        aux2.Children(1).Color = codcolor(7,:);

        aux2.Children(6).LineWidth = 1.5;
        aux2.Children(5).LineWidth = 1.5;
        aux2.Children(4).LineWidth = 1.5;
        aux2.Children(3).LineWidth = 1.5;
        aux2.Children(2).LineWidth = 1.5;
        aux2.Children(1).LineWidth = 1.5;

        if Best_pos(i,1) > 1
            aux2.Children(6).DisplayName = strcat("CEKFS"," (N=",sprintf('%.0f)',BestN(i,1)));
        else
            aux2.Children(6).DisplayName = "CEKF";
        end
        if Best_pos(i,2) > 1
            aux2.Children(5).DisplayName = strcat("CSEKFS"," (N=",sprintf('%.0f)',BestN(i,2)));
        else
            aux2.Children(5).DisplayName = "CSEKF";
        end
        if Best_pos(i,3) > 1
            aux2.Children(4).DisplayName = strcat("CUKFS"," (N=",sprintf('%.0f)',BestN(i,3)));
        else
            aux2.Children(4).DisplayName = "CUKF";
        end
        if Best_pos(i,4) > 1
            aux2.Children(3).DisplayName = strcat("CCKFS"," (N=",sprintf('%.0f)',BestN(i,4)));
        else
            aux2.Children(3).DisplayName = "CCKF";
        end
        aux2.Children(2).DisplayName = strcat("CPF"," (N=",sprintf('%.0f)',BestN(i,5)));
        aux2.Children(1).DisplayName = strcat("CEnKF"," (N=",sprintf('%.0f)',BestN(i,6)));



        aux2.OuterPosition(1) = (1-mod(j,2))/2;
        aux2.OuterPosition(2) = 1-(floor((j+1)/2))/ceil(nest/2);
        aux2.OuterPosition(3) = 0.5;
        aux2.OuterPosition(4) = 1/ceil(nest/2);
        xlabel('Time [h]', 'Interpreter','latex')
        ylabel(convertStringsToChars(Tagy(j)), 'Interpreter','latex');
        axis tight
        if j == 6
            aux2.Children(1).Parent.YLim =[-0.1, 0.1];
        end
    end

    ti_max = zeros(1,2);
    Margin = 0.02;
    for j = 1:nest
        aux2 = aux.Children(nest+1-j);
        switch j
            case 1
                if i < 3
                    ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+1-j-2).Position(1)]);
                else
                    ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+1-j-2).Position(1),aux.Children(nest+1-j-4).Position(1)]);
                end
                ti_max(2) = max([aux.Children(nest+1-j).Position(2),aux.Children(nest+1-j-1).Position(2)]);
                aux2.Position(3) = aux2.OuterPosition(3)+aux2.OuterPosition(1)-ti_max(1)-Margin;
                aux2.Position(4) = aux2.OuterPosition(4)+aux2.OuterPosition(2)-ti_max(2)-Margin;
                aux2.Position(1) = ti_max(1);
                aux2.Position(2) = ti_max(2);

            case 2
                if i < 3
                    ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+1-j-2).Position(1)]);
                else
                    ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+1-j-2).Position(1),aux.Children(nest+1-j-4).Position(1)]);
                end
                ti_max(2) = max([aux.Children(nest+1-j).Position(2),aux.Children(nest+1-j+1).Position(2)]);
                aux2.Position(3) = aux2.OuterPosition(3)+aux2.OuterPosition(1)-ti_max(1)-Margin;
                aux2.Position(4) = aux2.OuterPosition(4)+aux2.OuterPosition(2)-ti_max(2)-Margin;
                aux2.Position(1) = ti_max(1);
                aux2.Position(2) = ti_max(2);
            case 3
                if i < 3
                    ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+1-j+2).Position(1)]);
                else
                    ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+1-j-2).Position(1),aux.Children(nest+1-j+2).Position(1)]);
                end
                ti_max(2) = max([aux.Children(nest+1-j).Position(2),aux.Children(nest+1-j-1).Position(2)]);

                aux2.Position(3) = aux2.OuterPosition(3)+aux2.OuterPosition(1)-ti_max(1)-Margin;
                aux2.Position(4) = aux2.OuterPosition(4)+aux2.OuterPosition(2)-ti_max(2)-Margin;
                aux2.Position(1) = ti_max(1);
                aux2.Position(2) = ti_max(2);


                aux2.NextPlot = 'add';
                plot(aux2, Plant{2}.T, Plant{2}.Z(1,:), 'X');
                aux2.Children(1).LineWidth = 0.5;
                aux2.Children(1).Color = codcolor(1,:);
                aux2.Children(1).DisplayName = "Measurement";
                legend(aux2.Children([1,8:-1:2]),'Interpreter','latex')
                aux2.Legend.NumColumns = 1;
                aux2.Legend.FontSize = 6.5;
                aux2.Legend.Location = 'northwest';
                aux2.Legend.AutoUpdate = 'off';
                pause(0.01)
                delete(aux2.Children(1));

            case 4
                ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+2-j+2).Position(1)]);
                ti_max(2) = max([aux.Children(nest+1-j).Position(2),aux.Children(nest+2-j+1).Position(2)]);
                aux2.Position(3) = aux2.OuterPosition(3)+aux2.OuterPosition(1)-ti_max(1)-Margin;
                aux2.Position(4) = aux2.OuterPosition(4)+aux2.OuterPosition(2)-ti_max(2)-Margin;
                aux2.Position(1) = ti_max(1);
                aux2.Position(2) = ti_max(2);

            case 5
                ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+2-j+2).Position(1)]);
                ti_max(2) = max([aux.Children(nest+1-j).Position(2),aux.Children(nest+1-j-1).Position(2)]);
                aux2.Position(3) = aux2.OuterPosition(3)+aux2.OuterPosition(1)-ti_max(1)-Margin;
                aux2.Position(4) = aux2.OuterPosition(4)+aux2.OuterPosition(2)-ti_max(2)-Margin;
                aux2.Position(1) = ti_max(1);
                aux2.Position(2) = ti_max(2);


            case 6

                ti_max(1) = max([aux.Children(nest+1-j).Position(1),aux.Children(nest+2-j+2).Position(1)]);
                ti_max(2) = max([aux.Children(nest+1-j).Position(2),aux.Children(nest+1-j+1).Position(2)]);
                aux2.Position(3) = aux2.OuterPosition(3)+aux2.OuterPosition(1)-ti_max(1)-Margin;
                aux2.Position(4) = aux2.OuterPosition(4)+aux2.OuterPosition(2)-ti_max(2)-Margin;
                aux2.Position(1) = ti_max(1);
                aux2.Position(2) = ti_max(2);
        end
%         if i == 3
%             aux2.Position(3) = aux2.Position(3)+0.03;
%             aux2.Position(1) = aux2.Position(1)-0.03;
%         end
    end
    print(strcat(dir_save,sprintf("Const_Case%.0f_Best",i)),"-djpeg","-r300", '-image');
end

end

