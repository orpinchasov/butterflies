function [ full_neuron_firing_per_bin, angles ] = create_spike_count_and_angles_vector_ver1( period, T, G, Ang )
%CREATE_SPIKE_COUNT_AND_ANGLES_VECTOR Summary of this function goes here
%   Detailed explanation goes here

    global BEHAVIORAL_SAMPLE_RATE NEURONAL_SAMPLE_RATE BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN;
    
    number_of_neurons = max(G);
    
    % Create edges of entire sws period
    cluster_bin_edges = [0.5:1:number_of_neurons + 0.5];

    angles = [];
    full_neuron_firing_per_bin = [];

    for segment_index = 1:size(period, 1)
        current_period_start_behavior_sample_index = ceil(period(segment_index, 1) * BEHAVIORAL_SAMPLE_RATE);
        current_period_end_behavior_sample_index = floor(period(segment_index, 2) * BEHAVIORAL_SAMPLE_RATE);

        current_period_behavior_sample_edges = (current_period_start_behavior_sample_index - 0.5):BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN:(current_period_end_behavior_sample_index + 0.5);
        current_period_neuronal_sample_edges = current_period_behavior_sample_edges * NEURONAL_SAMPLE_RATE / BEHAVIORAL_SAMPLE_RATE;

        current_T = T(T > round(current_period_start_behavior_sample_index / BEHAVIORAL_SAMPLE_RATE * NEURONAL_SAMPLE_RATE) & ...
                      T < round(current_period_end_behavior_sample_index / BEHAVIORAL_SAMPLE_RATE * NEURONAL_SAMPLE_RATE));
        current_G = G(T > round(current_period_start_behavior_sample_index / BEHAVIORAL_SAMPLE_RATE * NEURONAL_SAMPLE_RATE) & ...
                      T < round(current_period_end_behavior_sample_index / BEHAVIORAL_SAMPLE_RATE * NEURONAL_SAMPLE_RATE));

        neuron_firing_per_bin = hist3([current_T current_G], 'Edges', {current_period_neuronal_sample_edges cluster_bin_edges});

        full_neuron_firing_per_bin = [full_neuron_firing_per_bin; neuron_firing_per_bin(1:end-1,1:end-1)];
        
        current_segment_start_angle_index = current_period_start_behavior_sample_index;
        current_segment_end_angle_index = current_segment_start_angle_index + ...
                                          (length(current_period_behavior_sample_edges) - 1) * BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN - 1;
        angles_during_segment = Ang(current_segment_start_angle_index:current_segment_end_angle_index);
        
        % TODO: Check here which is longer. the number of behavioral samples or
        % the neuronal samples per this session. if neuronal, then pad with
        % -1 in the angle measurements. if behavioral cut the extra angles
        % (we can't assume the neuronal data).
        
        % Bin angles
        angles_mat = reshape(angles_during_segment, [BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN size(neuron_firing_per_bin, 1) - 1])';
        angles_valid_mask_mat = (angles_mat ~= -1);
        angle_per_temporal_bin = mod(angle(sum(angles_valid_mask_mat .* exp(1i * angles_mat), 2)), 2 * pi);
        angle_per_temporal_bin(abs(sum(angles_valid_mask_mat .* exp(1i * angles_mat), 2)) == 0) = nan;
        angles = [angles; angle_per_temporal_bin];
    end    
end

