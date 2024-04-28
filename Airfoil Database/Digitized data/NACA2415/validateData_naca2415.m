%% Clear
clear; close all; clc
%% 
AFstr = 'NACA2415'; % Name of airfoil

% Get files within folder
fileList = dir([ AFstr '_*.csv']);

% Plot all data 
for i = 1:numel(fileList)
    data_L = readmatrix([fileList(i).folder '\' fileList(i).name]);
%     data_U = readmatrix([fileList(i).folder '\' fileList(i).name]);
    ReVal = str2double(extractBetween(fileList(i).name, 'Re','.csv'));%extractBetween(fileList(i).name, '_A','_');

    figure('units','normalized','outerposition',[0 0 1 1])
    t = tiledlayout(1,3);
    % Iterate thru Re number
    for M = 1:numel(data_L(1,:))-1
        if data_L(1, M+1) ~= data_L(1, M+1)
            % Check reading from same Mach number
            break
        end

        nexttile
        X_L = data_L(2:end,1);
        Y_L = data_L(2:end,1+M);
%         X_U = data_U(2:end,1);
%         Y_U = data_U(2:end,1+M);
    
        plot(X_L, Y_L,'o-')
%         hold on
%         plot(X_U, Y_U,'o-')
        
        
        xlabel('x/c')
        ylabel('C_p')
        title(['Re = ' num2str(data_L(1, M+1))])
        set(gca,'Ydir','reverse')
    end
    %leg = legend('Upper', 'Lower', 'Location','southeast');
    %leg.Layout.Tile = 'south';
    title(t, ['\alpha = ' num2str(ReVal)])
end
