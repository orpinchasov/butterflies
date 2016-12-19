function save_analysis_results( data_path, mouse_by_day_name, behavioral_state, reduced_data, angle_per_temporal_bin, spike_rate_mat_neuron_by_angle, estimated_head_direction_angle_per_sample_index )
    output_folder_name = [data_path '\' mouse_by_day_name '\output\' behavioral_state];
    mkdir(output_folder_name);

    save([output_folder_name '\reduced_data'], 'reduced_data');
    save([output_folder_name '\angle_per_temporal_bin'], 'angle_per_temporal_bin');
    save([output_folder_name '\spike_rate_mat_neuron_by_angle'], 'spike_rate_mat_neuron_by_angle');
    % When behavioral state is 'wake' the estimated head direction can be
    % used as sanity check against the real angle.
    save([output_folder_name '\estimated_head_direction_angle_per_sample_index'], 'estimated_head_direction_angle_per_sample_index');
end

