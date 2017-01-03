function [ spike_rate_mat_neuron_by_estimated_angle ] = calculate_spike_rate_neuron_by_estimated_angle( full_neuron_firing_rate, angle_per_temporal_bin )
%CALCULATE_SPIKE_RATE_NEURON_BY_ESTIMATED_ANGLE Summary of this function goes here
%   Detailed explanation goes here

    global SAMPLE_LIMIT NUMBER_OF_ANGLE_BINS CENTER_OF_ANGLE_BINS;
    
    number_of_neurons = size(full_neuron_firing_rate, 2);
    
    if length(angle_per_temporal_bin) > SAMPLE_LIMIT
        angle_per_temporal_bin = angle_per_temporal_bin(1:SAMPLE_LIMIT);
    end

    spike_rate_mat_neuron_by_estimated_angle = nan(number_of_neurons, NUMBER_OF_ANGLE_BINS);

    index_of_visualization_angle_per_temporal_bin = round(NUMBER_OF_ANGLE_BINS * mod(angle_per_temporal_bin, 2 * pi) / (2 * pi));
    index_of_visualization_angle_per_temporal_bin(index_of_visualization_angle_per_temporal_bin > NUMBER_OF_ANGLE_BINS) = NUMBER_OF_ANGLE_BINS;
    index_of_visualization_angle_per_temporal_bin(index_of_visualization_angle_per_temporal_bin < 1) = 1;
    index_of_visualization_angle_per_temporal_bin(isnan(index_of_visualization_angle_per_temporal_bin)) = NUMBER_OF_ANGLE_BINS + 1;
    
    number_of_frames_per_angular_velocity = hist(angle_per_temporal_bin, CENTER_OF_ANGLE_BINS);

    for angle_bin_index = 1:NUMBER_OF_ANGLE_BINS;
        current_bin_neurons_number_of_spikes = sum(full_neuron_firing_rate(index_of_visualization_angle_per_temporal_bin == angle_bin_index, :));

        spike_rate_mat_neuron_by_estimated_angle(:, angle_bin_index) = current_bin_neurons_number_of_spikes ./ number_of_frames_per_angular_velocity(angle_bin_index);
    end

end

