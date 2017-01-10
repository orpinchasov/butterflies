% Test various characteristics of specific points on the scatter plot. For
% example, test whether out-of-order points seem to have lower firing
% rates.
% We enable the choosing of specific points here using 'ginput'.

%% Plot the rapid firing neurons to check their positions on the scatter plot
significant_neurons = [25 43 44 46 49 54 55 59 60];

%neuron_activity_indices = filtered_neuron_firing(:, 3) > 0;

%figure;
%hold on;
%scatter(reduced_data(:, 2), reduced_data(:, 3), 5, '.k');
%scatter(reduced_data(neuron_activity_indices, 2), reduced_data(neuron_activity_indices, 3), 20, '.r');

for i = 1:length(full_neuron_firing_per_bin)
    neuron_activity_indices = filtered_neuron_firing(:, i) > 0;
    
    figure;
    hold on;
    scatter(reduced_data(:, 2), reduced_data(:, 3), 5, '.k');
    scatter(reduced_data(neuron_activity_indices, 2), reduced_data(neuron_activity_indices, 3), 20, '.r');
    
    saveas(gca, ['E:\or\results\fig4_ver3\neurons_position_on_ring\' num2str(i) '.jpg'], 'jpg');
end

%% Choose 4 points on the reduced data plot and get the indices of all points in the polygon created by these points
figure;
scatter(reduced_data(:, 2), reduced_data(:, 3), 5, cmap2(index_of_visualization_angle_per_temporal_bin, :), 'fill');
[x y] = ginput(4);

%%
points_in_polygon = inpolygon(reduced_data(:, 2), reduced_data(:, 3), x, y);

figure;
scatter(reduced_data(points_in_polygon, 2), reduced_data(points_in_polygon, 3), '.');

figure;
hold on;
plot(amount_of_movement);

%% Plot scattering of REM events overlayed with total number of active neurons per time frame.
cmap_number_of_active_neurons = hsv(max(number_of_active_neurons));

figure; scatter(reduced_data(:, 2), reduced_data(:, 3), 20, cmap_number_of_active_neurons(number_of_active_neurons(number_of_active_neurons > NUMBER_OF_ACTIVE_NEURONS_THRESHOLD), :), '.'); colorbar;

figure; scatter3(reduced_data(:, 2), reduced_data(:, 3), number_of_active_neurons(number_of_active_neurons > NUMBER_OF_ACTIVE_NEURONS_THRESHOLD), 20, '.');
