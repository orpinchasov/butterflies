function [ spike_rate_mat_neuron_by_angular_velocity, index_of_visualization_angular_velocity_per_temporal_bin ] = calculate_spike_rate_neuron_by_angular_velocity( full_neuron_firing_rate, angle_per_temporal_bin )
    global SAMPLE_LIMIT NUMBER_OF_ANGULAR_VELOCITY_BINS MAX_ANGULAR_VELOCITY MIN_ANGULAR_VELOCITY;
    
    number_of_neurons = size(full_neuron_firing_rate, 2);
    
    if length(angle_per_temporal_bin) > SAMPLE_LIMIT
        angle_per_temporal_bin = angle_per_temporal_bin(1:SAMPLE_LIMIT);
    end
        
    %valid_angle_during_awake = angle_during_awake(angle_during_awake ~= -1);
    %valid_angle_during_awake_histogram = hist(valid_angle_during_awake, CENTER_OF_ANGLE_BINS);

    spike_rate_mat_neuron_by_angular_velocity = nan(number_of_neurons, NUMBER_OF_ANGULAR_VELOCITY_BINS);
    
    angular_velocity = mod(diff(angle_per_temporal_bin) + pi, 2 * pi) - pi;
    angular_velocity = [angular_velocity; angular_velocity(end)];
        
    index_of_visualization_angular_velocity_per_temporal_bin = round(NUMBER_OF_ANGULAR_VELOCITY_BINS * ((angular_velocity - MIN_ANGULAR_VELOCITY) / (MAX_ANGULAR_VELOCITY - MIN_ANGULAR_VELOCITY)));
    index_of_visualization_angular_velocity_per_temporal_bin(index_of_visualization_angular_velocity_per_temporal_bin > NUMBER_OF_ANGULAR_VELOCITY_BINS) = NUMBER_OF_ANGULAR_VELOCITY_BINS;
    index_of_visualization_angular_velocity_per_temporal_bin(index_of_visualization_angular_velocity_per_temporal_bin < 1) = 1;
    index_of_visualization_angular_velocity_per_temporal_bin(isnan(index_of_visualization_angular_velocity_per_temporal_bin)) = NUMBER_OF_ANGULAR_VELOCITY_BINS + 1;
    
    number_of_frames_per_angular_velocity = hist(index_of_visualization_angular_velocity_per_temporal_bin, 0.5:1:NUMBER_OF_ANGULAR_VELOCITY_BINS + 1.5);

    for angular_velocity_bin_index = 1:NUMBER_OF_ANGULAR_VELOCITY_BINS;
        current_bin_neurons_number_of_spikes = sum(full_neuron_firing_rate(index_of_visualization_angular_velocity_per_temporal_bin == angular_velocity_bin_index, :));

        spike_rate_mat_neuron_by_angular_velocity(:, angular_velocity_bin_index) = current_bin_neurons_number_of_spikes ./ number_of_frames_per_angular_velocity(angular_velocity_bin_index);
    end
    
end

