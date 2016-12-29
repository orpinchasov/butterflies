function save_analysis_results( data_path, mouse_by_day_name, brain_region, behavioral_state, full_neuron_firing_per_bin, full_reduced_data, angle_per_temporal_bin, spike_rate_mat_neuron_by_angle, estimated_head_direction_angle_per_sample_index )
    switch brain_region
        case 1
            brain_region_str = 'thalamus';
        case 2
            brain_region_str = 'subiculum';
        case 3
            brain_region_str = 'hippocampus';
        case 4
            brain_region_str = 'prefrontal';
        case 5
            brain_region_str = 'all';
    end

    output_folder_name = [data_path '\' mouse_by_day_name '\output\' brain_region_str '\' behavioral_state];
    mkdir(output_folder_name);

    save([output_folder_name '\full_neuron_firing_per_bin'], 'full_neuron_firing_per_bin');
    save([output_folder_name '\full_reduced_data'], 'full_reduced_data');
    save([output_folder_name '\angle_per_temporal_bin'], 'angle_per_temporal_bin');
    save([output_folder_name '\spike_rate_mat_neuron_by_angle'], 'spike_rate_mat_neuron_by_angle');
    % When behavioral state is 'wake' the estimated head direction can be
    % used as sanity check against the real angle.
    save([output_folder_name '\estimated_head_direction_angle_per_sample_index'], 'estimated_head_direction_angle_per_sample_index');
end

