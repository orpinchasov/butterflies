function [ full_neuron_firing_per_bin, reduced_data, angle_per_temporal_bin ] = analyze_neuronal_and_behavioral_data(T, G, Ang, period)
%ANALYZE_MOUSE_AND_STATE Summary of this function goes here
%   Detailed explanation goes here

    % TODO: Move all constants to a single location or change
    % them from being constants.
    % p is the percentage of closest neighbors to include in the near neighbors
    P_NEIGHBORS_VEC = [0.075 / 30 0.075];
    NUMBER_OF_REDUCED_DIMENSIONS_VEC = [10 10];
    SAMPLE_LIMIT = 20000;

    [full_neuron_firing_per_bin, angle_per_temporal_bin] = create_spike_count_and_angles_vector_ver1(period, T, G, Ang);
    
    % Possible post-processing of the data

    % Binarize the data of the number of spiking per time frame (any number of
    % spikes is equal to binary True)
    full_neuron_firing_per_bin = 1.0*(~~full_neuron_firing_per_bin);
    % Threshold minimum number of spikes at 2
    %full_neuron_firing_per_bin = 1.0*(full_neuron_firing_per_bin > 1);
    % Reduce long distrances
    %full_neuron_firing_per_bin = log(full_neuron_firing_per_bin + 1);

    % Truncate the data to avoid exceeding maximum array size
    if size(full_neuron_firing_per_bin, 1) > SAMPLE_LIMIT
        full_neuron_firing_per_bin = full_neuron_firing_per_bin(1:SAMPLE_LIMIT, :);
        angle_per_temporal_bin = angle_per_temporal_bin(1:SAMPLE_LIMIT);
    end
    
    reduced_data = create_reduced_data(full_neuron_firing_per_bin, P_NEIGHBORS_VEC, NUMBER_OF_REDUCED_DIMENSIONS_VEC);
end

