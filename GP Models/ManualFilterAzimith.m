

%data = readtable('demo_outboard_CEREAL.csv');
data = readtable('demo_outboard_CEREAL_edited.csv');
head(data)
azim_list = unique(data.azim)

%Check for Interpolation needed -- Not needed
% slice = data(data.azim == azim_val,:);
% r = numel(unique(slice.r))
% z = numel(unique(slice.z))

%for n = 1:length(azim_list)
n = 6
    azim_val = azim_list(n);
    slice = data(data.azim == azim_val,:);
    r_unique = unique(slice.r);
    z_unique = unique(slice.z);

% Initialize an empty grid
    VmagMatrix = nan(length(z_unique), length(r_unique));

    for i = 1:length(r_unique)
        for j = 1:length(z_unique)
            idx = slice.r == r_unique(i) & slice.z == z_unique(j);
            if any(idx)
                VmagMatrix(j,i) = slice.Vmag(idx);
            end
        end
    end

    contourf(r_unique, z_unique, VmagMatrix, 30, 'LineColor', 'none');
    colorbar;
    xlabel('r'); ylabel('z');
    title(['Azimuth Slice: ', num2str(azim_list(n)), '°']);
    axis equal tight;
    % r_unique = unique(slice.r);
    % z_unique = unique(slice.z);
    % Vmag_matrix = reshape(slice.Vmag,[length(r_unique),length(z_unique)]);
    % figure;
    % contourf(r_unique,z_unique,Vmag_matrix,30,'LineColor','none');
    % colorbar;
    % xlabel('r');
    % ylabel('z');
    % title(['Azimuth Slice: ',num2str(azim_val),' deg']);
    % axis equal tight;
%end





