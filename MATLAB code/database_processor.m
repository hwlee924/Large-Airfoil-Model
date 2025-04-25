% Script description: This is an automation code for updating all the csv
% files their respective folders for their uncertainty parameters (Database
% management)
% Author: Ansh Atul Mishra
%%------------------------------------------------------------------------%%
clear all; close all; clc;
folderPath = '/Users/anshmishra/Documents/CEREAL/CEREAL_LOCAL_REP/ASPIRE/Airfoils/NACA 23012/';  % Change this to your actual folder path
files = dir(folderPath);
csvFiles = dir(fullfile(folderPath, '*.csv'));

for i = 1:length(files)
    if endsWith(files(i).name, '.json', 'IgnoreCase', true)
        jsonFileName = files(i).name;
        fullFilePath_json = fullfile(folderPath, jsonFileName);
        jsonText = fileread(fullFilePath_json);
        data_json = jsondecode(jsonText);
        uncertainty_params = {'x', 'cp', 'alpha', 'mach'};
        for j = 1:length(uncertainty_params)
            param = uncertainty_params{j};
            if ~isempty(data_json.uncertainty.(param))
                for k = 1:length(csvFiles)
                    if endsWith(csvFiles(k).name, 'A.csv')
                        csv_file = fullfile(folderPath, csvFiles(k).name);
                        csv_data = readcell(csv_file);
                        uncertainty_param_column = cell(length(csv_data), 1);
                        uncertainty_param_column{1} = uncertainty_params{j};
                        for m = 2:length(csv_data)
                            uncertainty_param_column{m} = data_json.uncertainty.(param);
                        end
                        csv_data = [csv_data uncertainty_param_column];
                        csv_data{1,1} = '';
                        writecell(csv_data, csv_file);
                    end
                end
            else
                fprintf('No uncertainty values \n');
            end
        end
    end
end