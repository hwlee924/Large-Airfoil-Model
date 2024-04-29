clear; close all; clc

finalData_X = table;
finalData_Y = table;

coord1 = readmatrix('.\NACA64A_series\Converted\NACA64A006\NACA64A006_coordinates.csv');

ref_x_U = flip([0 0.5 0.75 1.25 2.5 5.0 7.5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]'/100); %flip(1-cos(linspace(0,pi/2,30)))';
ref_x_L = [0 0.5 0.75 1.25 2.5 5.0 7.5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]'/100;%1-cos(linspace(0,pi/2,30))';

afStrList = {'NACA64A006','NACA64A010','NACA64A406','NACA64A410', ...
    'CRA309-A', 'CRA309-B', 'NACA0012', 'NLRQE0.11-0.75-1.375', ...
    'NLR7301', 'RAE2822', 'SSCA09'  ...
    'MBBA3', 'Supercritical9a', 'SC1095', 'WT180', ...
    'NACA0015', 'NACA23012', 'NACA63-415', 'NACA0011', ...
    'NACA4412','DSMA523', 'BAC1', 'OLSTAAT',}% ...
    %'SSCA09','SSCA07'};
noiseTypeList = {'constant','constant','constant','constant',...
    'constant', 'constant', 'constant', 'constant', ...
    'constant','constant', 'constant', ...
    'relativeMax', 'relativeMax', 'constant', 'constant', ...
    'constant', 'relative','constant', 'constant', ...
    'constant', 'relativeMax', 'relative', 'constant'}%, ...
   % 'constant', 'constant'};
symmetryList = {'symmetric', 'symmetric', 'cambered', 'cambered', ...
    'cambered', 'cambered', 'symmetric', 'symmetric', ...
    'cambered', 'cambered', 'cambered', ...
    'cambered', 'cambered', 'cambered', 'cambered', ...
    'symmetric', 'cambered', 'cambered', 'symmetric', ...
    'cambered', 'cambered', 'cambered','symmetric'}%,...
   % 'cambered', 'cambered'};
supercriticalList = {'', '', '', '', ...
    '', '', '', 'supercritical', ...
    'supercritical', 'supercritical', 'supercritical', '', ...
    'supercritical', 'supercritical', '', '', ...
    '', '', '', '', ...
    '','supercritical','',''}%,...
 %   '', ''};
noiseList = [0.01, 0.01, 0.01, 0.01, ...
    0.01, 0.01, 0.05, 0.02, ...
    0.5/100, 0.02, 0.0064, 0.01, ...
    1/100, 2e-2, 0.01, 0.01, ...
    1/100, 2.5e-2, 0.01, 0.0144, ...
    0.03, 15e-3, 25e-3, 0.01, ...
    0.01, 0.01];

%% 
for i = 1:numel(afStrList)
    afStrList{i}
    [temp_x, temp_y] = process_(afStrList{i}, noiseTypeList{i}, symmetryList{i}, supercriticalList{i}, noiseList(i), [ref_x_U, ref_x_L]);
    finalData_X = [finalData_X; temp_x];
    finalData_Y = [finalData_Y; temp_y];
end
%% 
bools2 = ~isnan(finalData_Y.Cp);
finalData_X = finalData_X(bools2, :);
finalData_Y = finalData_Y(bools2, :);

finalData = [finalData_X, finalData_Y];
finalData.noise(finalData.noise<1e-3) = 1e-3;

saveVar = true;
if saveVar 
    writetable(finalData, 'demo.csv')
end
%% Conversion Code
function [data_X, data_Y] = process_(af_string, noise_type, symmetryList, supercriticalList, noise, ref_x)
data_X = table;
data_Y = table;

coords = readmatrix(['.\Paper\' af_string '\' af_string '_coordinates.csv']);
fileList = dir(['.\' af_string '\' af_string '*.csv']);

ind_h = find(diff(sign(diff(coords(1:end,1))))); % Detect change in direction - i.e. intersection b/n upper and lower surf in coordinates
ind_h = ind_h(1)+1;

for i = 1:numel(fileList)-1 % Run thru files except coordinates
    data = readmatrix([fileList(i).folder '\' fileList(i).name]);
    ind_hh = find(diff(sign(diff(data(2:end,1))))); % intersection b/n upper and lower surf in pressure file
    ind_hh = ind_hh(1)+2; % correction


    %     [~, ind] = min(abs(data(1,:)-targetM)); % index of given Mach number
    for ind = 2:numel(data(1,:))
        % Pressure data and its locations
        data_xU = -data(2:ind_hh, 1);
        data_xL = data(ind_hh+1:end, 1);
        data_x  = [data_xU; data_xL];

        data_U = data(2:ind_hh, ind);
        data_L = data(ind_hh+1:end, ind); % lmao spaghetti
        data_cp = [data_U; data_L];

        ref_x_U = ref_x(:,1);
        ref_x_L = ref_x(:,2);

        temp_x_u = interp1(coords(1:ind_h, 1), coords(1:ind_h,2), ref_x_U, 'linear','extrap');
        temp_x_l = interp1(coords(ind_h+1:end,1), coords(ind_h+1:end,2), ref_x_L, 'linear','extrap');

        for j = 1:numel(data_x)
            xLoc = data_x(j,1);
            M = data(1, ind);
            Cp = data_cp(j);
            supercritical = supercriticalList;
            symmetry = symmetryList;
            switch noise_type
                case 'constant'
                    noise_val = noise;
                case 'relative'
                    if abs(Cp) * noise > 0.01
                        noise_val = abs(Cp) * noise;
                    else
                        noise_val = 0.01;
                    end
                case 'relativeMax'
                    if max(abs(data_cp)) * noise > 0.01
                        noise_val = max(abs(data_cp)) * noise;
                    else 
                        noise_val = 0.01
                    end
            end
            data_X = [data_X; cell2table({ref_x_U', ref_x_L', temp_x_u', temp_x_l', getAoA([fileList(i).folder '\' fileList(i).name]), xLoc, M, noise_val, af_string, symmetry, supercritical})];
            data_Y = [data_Y; table(Cp,'VariableNames',{'Cp'})]; %[data_Y; Cp];
        end
    end
end
data_X.Properties.VariableNames = {'x_u','x_l','z_u','z_l','alpha','xc','M','noise','af','symmetry', 'supercritical'};
end

function out = getAoA(str)
str2 = extractBetween(str, '_A', '_' );
if str2{1}(1) == 'm'
    str2{1}(1) = '-';
end
out = str2double(str2{1});
end

