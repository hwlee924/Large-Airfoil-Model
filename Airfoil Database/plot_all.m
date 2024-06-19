%% Clear
clear; close all; clc;
%% Inputs 
directory_str = '.\Digitized Data\SC 1094 R8\';
cutoff_ind = 23;

%% Re-sort data 
file_str = '*.csv';
file_list = dir([directory_str, file_str]);

A_arr = zeros(numel(file_list)-1, 1);
M_arr = zeros(numel(file_list)-1, 1);

for i = 1:numel(file_list)-1
    % Get angle
    a = extractBetween(file_list(i).name, '_A', '_M');
    a = a{1};
    if a(1) == 'm'
        a = -str2double(a(2:end));
    else 
        a = str2double(a);
    end
    
    % Get Mach
    m = extractBetween(file_list(i).name, '_M', '_Re');
    m = str2double(m{1});
    
    A_arr(i) = a;
    M_arr(i) = m;
end
[AM_arr, idx] = sortrows([A_arr, M_arr], [2, -1], {'ascend', 'ascend'});

%% 
for i = 1:numel(file_list)-1
    figure
    data = readmatrix([directory_str file_list(idx(i)).name]);
    
    plot(data(2:cutoff_ind,1), data(2:cutoff_ind,2), 's-', LineWidth=2, displayName='Upper')
    hold on
    plot(data(cutoff_ind:end, 1), data(cutoff_ind:end,2),'o-', LineWidth=2, displayName='Lower')
    xlim([-0.05, 1.05])
    xlabel('x/c')
    ylabel('Cp')
    legend()
    set(gca, 'YDir','reverse')
    title(['M:', num2str(AM_arr(i,2)), ' A:', num2str(AM_arr(i,1)) ])
    drawnow
end