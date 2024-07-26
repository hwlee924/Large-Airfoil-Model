clear; close all; clc

finalData_X = table;
finalData_Y = table;

%% Read airfoil list 
airfoil_list_name = 'list_airfoils.xlsx';
opts = detectImportOptions(airfoil_list_name, 'NumHeaderLines', 0); 
data_table = readtable(airfoil_list_name, opts);
coord1 = readmatrix('.\Digitized data\NACA 64A006\NACA 64A006_coordinates.csv');

%%
ref_x_U = flip([0 0.25 0.75, 1.0 1.5 2.0 2.5 5.0 7.5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]'/100); 
ref_x_L = [0 0.25 0.75, 1.0 1.5 2.0 2.5 5.0 7.5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]'/100; 


% REplace this later
noiseTypeList = {'constant', 'constant', 'constant', 'constant', 'relative'   , 'unknown', ...
             'constant', 'constant', 'unknown' , 'unknown' , 'unknown'    , 'unknown', ...
             'constant', 'constant', 'relative', 'constant'   , 'relativeMax', 'relativeMax', ...
             'unknown' , 'unknown' , 'constant', 'relative', 'unknown'    , 'unknown', ...
             'unknown' , 'relative', 'unknown' , 'unknown' , 'unknown'    , 'relative', ...
             'constant', 'unknown' , 'unknown' , 'unknown' , 'unknown'    , 'unknown' , ...
             'unknown' , 'unknown',  'unknown' , 'unknown' , 'unknown'    , 'unknown' , ...
             'unknown' , 'unknown',  'unknown' , 'unknown' , 'unknown'    , 'unknown' , ...
             'unknown', 'constant'};

noiseList = [0.01, 0.01,   0.01, 0.01, 2/100, 0.0, ...
             0.03, 0.0144, 0.0,  0.0,  0.0  , 0.0, ...
             0.05, 0.02, 0.5/100, 0.02, 1/100, 2/100, ...
             0.0, 0.0, 0.03, 0.25/100, 0.0, 0.0, ...
             0.0, 1/100, 0.0, 0.0, 0.0, 1/100, ...
             0.08, 0.0, 0.0, 0.0, 0.0, 0.0, ...
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
             0.0, 0.0];
%% Verify airfoil geoms
% for i = 1:numel(data_table.Airfoil)
%     if data_table.Use{i} == 'Y'
%         disp(data_table.Airfoil{i})
%         check_airfoil_geom(i, data_table, [ref_x_U, ref_x_L]);
%     else
%     end
%     
% end
%% 
num_cases = 0;
counter = 1; 
num_case = 1;
for i = 1:numel(data_table.Airfoil)
    if data_table.Use{i} == 'Y'
        disp(data_table.Airfoil{i})
        [temp_x, temp_y, nc] = process_(i, data_table, noiseTypeList{counter}, noiseList(counter), [ref_x_U, ref_x_L], num_cases);
        finalData_X = [finalData_X; temp_x];
        finalData_Y = [finalData_Y; temp_y];
        num_cases = num_cases + nc;
        counter = counter + 1;
        disp(nc)
    else
    end
%     [temp_x, temp_y, nc] = process_(afStrList{i}, noiseTypeList{i}, symmetryList{i}, supercriticalList{i}, noiseList(i), [ref_x_U, ref_x_L]);
%     finalData_X = [finalData_X; temp_x];
%     finalData_Y = [finalData_Y; temp_y];
%     num_ = num_ + nc;
    
end
disp(num_cases)
%% 
bools2 = ~isnan(finalData_Y.Cp);
finalData_X = finalData_X(bools2, :);
finalData_Y = finalData_Y(bools2, :);
bools3 = ~isnan(finalData_X.xc);
finalData_X = finalData_X(bools3, :);
finalData_Y = finalData_Y(bools3, :);

finalData = [finalData_X, finalData_Y];
finalData.yc = real(finalData.yc);
finalData.noise(finalData.noise<1e-3) = 1e-3;

saveVar = true;
if saveVar 
    writetable(finalData, '20240626_corrections.csv')
end
%% Conversion Code
function [data_X, data_Y, num_cases] = process_(idx, reference_table, noise_type, noise, ref_x, start_num) %process_(af_string, noise_type, symmetryList, supercriticalList, noise, ref_x)
num_cases = 0; % Total number of unique cases within this particular airfoil
case_num = start_num; % number of case for the entire data set

data_X = table;
data_Y = table;

af_string = reference_table.Airfoil{idx};

coords = readmatrix(['.\Digitized data\' af_string '\' af_string '_coordinates.csv']);
fileList = dir(['.\Digitized data\' af_string '\' af_string '*.csv']);

ind_h = find(diff(sign(diff(coords(1:end,1))))); % Detect change in direction - i.e. intersection b/n upper and lower surf in coordinates
ind_h = ind_h(1)+1;

for i = 1:numel(fileList)-1 % Run thru files except coordinates
    data = readmatrix([fileList(i).folder '\' fileList(i).name]);
    ind_hh = find(diff(sign(diff(data(2:end,1))))); % intersection b/n upper and lower surf in pressure file
    ind_hh = ind_hh(1)+2; % correction


    %     [~, ind] = min(abs(data(1,:)-targetM)); % index of given Mach number
    for ind = 2:numel(data(1,:))
        num_cases = num_cases + 1;
        case_num = case_num + 1;
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
%             xLoc = abs(data_x(j,1));
%             ang = acos(2*xLoc-1);
%             yLoc_convert = 0.5*sign(-data_x(j,1))*sin(ang);
            xLoc = abs(data_x(j,1))*2-1;
            ang = acos(xLoc);
            yLoc_convert = sign(-data_x(j,1))*sin(ang);
%             ang = asin(xLoc);
%             yLoc_convert = sign(-data_x(j,1))*cos(ang);
            M = data(1, ind);
            Cp = data_cp(j);
            
            af_type = reference_table.Usage{idx};
            af_family = reference_table.AirfoilFamily{idx};
            supercritical = reference_table.Supercritical{idx}; 
            symmetry = reference_table.Symmetry{idx}; 
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
                        noise_val = 0.01;
                    end
                case 'unknown'
                    noise_val = 0.01;
            end
            temp_data_x = cell2table({ref_x_U', ref_x_L', temp_x_u', temp_x_l', getAoA([fileList(i).folder '\' fileList(i).name]), xLoc, yLoc_convert, M, noise_val, af_string, symmetry, supercritical, af_type, af_family, case_num});
            data_X = [data_X; temp_data_x];
            data_Y = [data_Y; table(Cp,'VariableNames',{'Cp'})]; %[data_Y; Cp];
        end
        
        
    end
end
data_X.Properties.VariableNames = {'x_u','x_l','z_u','z_l','alpha','xc','yc','M','noise','af','symmetry', 'supercritical', 'usage','family', 'case'};
% figure
% plot(data_X.xc, data_X.yc,'o-')
% drawnow
% plot(ref_x_U, temp_x_u)
% hold on
% plot(ref_x_L, temp_x_l)
% xlabel('x/c')
% ylabel('z/c')
% title(af_string)
% drawnow
end

function out = check_airfoil_geom(idx, reference_table, ref_x)
af_string = reference_table.Airfoil{idx};
coords = readmatrix(['.\Digitized data\' af_string '\' af_string '_coordinates.csv']);
fileList = dir(['.\Digitized data\' af_string '\' af_string '*.csv']);

ind_h = find(diff(sign(diff(coords(1:end,1))))); % Detect change in direction - i.e. intersection b/n upper and lower surf in coordinates
ind_h = ind_h(1)+1;

for i = 1:numel(fileList)-1 % Run thru files except coordinates
    data = readmatrix([fileList(i).folder '\' fileList(i).name]);
    ind_hh = find(diff(sign(diff(data(2:end,1))))); % intersection b/n upper and lower surf in pressure file
    ind_hh = ind_hh(1)+2; % correction



    ref_x_U = ref_x(:,1);
    ref_x_L = ref_x(:,2);

    temp_x_u = interp1(coords(1:ind_h, 1), coords(1:ind_h,2), ref_x_U, 'linear','extrap');
    temp_x_l = interp1(coords(ind_h+1:end,1), coords(ind_h+1:end,2), ref_x_L, 'linear','extrap');
end
figure
plot(ref_x_U, temp_x_u, 'o-')
hold on
plot(ref_x_L, temp_x_l, 'o-')
title(af_string)
out = 0;
end

function out = getAoA(str)
str2 = extractBetween(str, '_A', '_' );
if str2{1}(1) == 'm'
    str2{1}(1) = '-';
end
out = str2double(str2{1});
end
