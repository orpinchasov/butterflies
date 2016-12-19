function [ reduced_data, angle_per_temporal_bin, spike_rate_mat_neuron_by_angle, estimated_head_direction_angle_per_sample_index ] = load_analysis_results(data_path, mouse_by_day_name, behavioral_state)
    input_folder_name = [data_path '\' mouse_by_day_name '\output\' behavioral_state];
    
    load([input_folder_name '\reduced_data']);
    load([input_folder_name '\angle_per_temporal_bin']);
    load([input_folder_name '\spike_rate_mat_neuron_by_angle']);
    % When behavioral state is 'wake' the estimated head direction can be
    % used as sanity check against the real angle.
    load([input_folder_name '\estimated_head_direction_angle_per_sample_index']);
end

