%% Clear
clear; close all; clc
%% 
AFstr = 'NACA0011'; % Name of airfoil

% Get files within folder
fileList = dir([ AFstr '_*.csv']);

% Plot all data 
for i = 1:round((numel(fileList)-1)/2)
    data_L = readmatrix([fileList(2*i-1).folder '\' fileList(2*i-1).name]);
    data_U = readmatrix([fileList(2*i).folder '\' fileList(2*i).name]);
    angleVal = extractBetween(fileList(2*i-1).name, '_A','_');
    if angleVal{1}(1) == 'm'
        angleVal = -str2double(angleVal{1}(2:end));
    else
        angleVal = str2double(angleVal);
    end
    figure('units','normalized','outerposition',[0 0 1 1])
    t = tiledlayout(1,1);
    % Iterate thru Re number
    for M = 1:numel(data_L(1,:))-1
        if data_L(1, M+1) ~= data_U(1, M+1)
            % Check reading from same Mach number
            break
        end

        nexttile
        X_L = data_L(2:end,1);
        Y_L = data_L(2:end,1+M);
        X_U = data_U(2:end,1);
        Y_U = data_U(2:end,1+M);
    
        plot(X_L, Y_L,'o-')
        hold on
        plot(X_U, Y_U,'o-')
        
        set(gca,'Ydir','reverse')
        xlabel('x/c')
        ylabel('C_p')
        title(['Re = ' num2str(data_L(1, M+1))])
    end
    %leg = legend('Upper', 'Lower', 'Location','southeast');
    %leg.Layout.Tile = 'south';
    title(t, ['\alpha = ' num2str(angleVal) ' deg'])
end
