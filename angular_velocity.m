% [spike_rate_mat_neuron_by_angular_velocity index_of_visualization_angular_velocity_per_temporal_bin] = calculate_spike_rate_neuron_by_angular_velocity(full_neuron_firing_per_bin, angle_per_temporal_bin);
cmap_angular_velocity = jet(NUMBER_OF_ANGULAR_VELOCITY_BINS / 2);
cmap_angular_velocity = [cmap_angular_velocity; [0 0 0]];

index_of_visualization_abs_angular_velocity_per_temporal_bin = ceil(abs(index_of_visualization_angular_velocity_per_temporal_bin - (NUMBER_OF_ANGULAR_VELOCITY_BINS / 2 + 0.5)));
index_of_visualization_abs_angular_velocity_per_temporal_bin(index_of_visualization_angular_velocity_per_temporal_bin == NUMBER_OF_ANGULAR_VELOCITY_BINS + 1) = NUMBER_OF_ANGULAR_VELOCITY_BINS / 2 + 1;

figure;
scatter3(reduced_data(:, 2), reduced_data(:, 3), reduced_data(:, 4), 20, cmap_angular_velocity(index_of_visualization_abs_angular_velocity_per_temporal_bin, :), 'fill');

%% Calculate angular velocity
NUMBER_OF_ANGULAR_VELOCITY_BINS = 16;
MAX_ANGULAR_VELOCITY = 0.25;
MIN_ANGULAR_VELOCITY = -0.25;

index_of_visualization_angular_velocity_per_temporal_bin = round(NUMBER_OF_ANGULAR_VELOCITY_BINS * ((angular_velocity - MIN_ANGULAR_VELOCITY) / (MAX_ANGULAR_VELOCITY - MIN_ANGULAR_VELOCITY)));
index_of_visualization_angular_velocity_per_temporal_bin(index_of_visualization_angular_velocity_per_temporal_bin > NUMBER_OF_ANGULAR_VELOCITY_BINS) = NUMBER_OF_ANGULAR_VELOCITY_BINS;
index_of_visualization_angular_velocity_per_temporal_bin(index_of_visualization_angular_velocity_per_temporal_bin < 1) = 1;
index_of_visualization_angular_velocity_per_temporal_bin(isnan(index_of_visualization_angular_velocity_per_temporal_bin)) = NUMBER_OF_ANGULAR_VELOCITY_BINS + 1;