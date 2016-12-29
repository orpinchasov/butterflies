% Import project-wide constants
constants

ALL_MOUSE_NAMES = {
    {'Mouse12-120806', [1 3 4]}, ...
    {'Mouse12-120807', [1 3 4]}, ...
    {'Mouse12-120808', [1 3 4]}, ...
    {'Mouse12-120809', [1 3 4]}, ...
    {'Mouse12-120810', [1 3 4]}, ...
    {'Mouse17-130125', [1 3]}, ...
    {'Mouse17-130128', [1 3]}, ...
    {'Mouse17-130129', [1 3]}, ...
    {'Mouse17-130130', [1 3]}, ...
    {'Mouse17-130131', [1 3]}, ...
    {'Mouse17-130201', [1 3]}, ...
    {'Mouse17-130202', [1 3]}, ...
    {'Mouse17-130203', [1 3]}, ...
    {'Mouse17-130204', [1 3]}, ...
    {'Mouse20-130514', [1]}, ...
    {'Mouse20-130515', [1]}, ...
    {'Mouse20-130516', [1]}, ...
    {'Mouse20-130517', [1]}, ...
    {'Mouse20-130520', [1]}, ...
    {'Mouse24-131213', [1 2]}, ...
    {'Mouse24-131216', [1 2]}, ...
    {'Mouse24-131217', [1 2]}, ...
    {'Mouse24-131218', [1 2]}, ...
    {'Mouse25-140123', [1 2]}, ...
    {'Mouse25-140124', [1 2]}, ...
    {'Mouse25-140128', [1 2]}, ...
    {'Mouse25-140129', [1 2]}, ...
    {'Mouse25-140130', [1 2]}, ...
    {'Mouse25-140131', [1 2]}, ...
    {'Mouse25-140203', [1 2]}, ...
    {'Mouse25-140204', [1 2]}, ...
    {'Mouse25-140205', [1 2]}, ...
    {'Mouse25-140206', [1 2]}, ...
    {'Mouse28-140310', [1 2]}, ...
    {'Mouse28-140311', [1 2]}, ...
    {'Mouse28-140312', [1 2]}, ...
    {'Mouse28-140313', [1 2]}, ...
    {'Mouse28-140317', [1 2]}, ...
    {'Mouse28-140318', [1 2]}
    };

%% Run with different brain regions

for mouse = ALL_MOUSE_NAMES
    mouse_name = mouse{1}{1};
    
    fprintf('Working on mouse %s\n', mouse_name);
    
    for brain_region = mouse{1}{2}
        fprintf('Working on brain region %d\n', brain_region);
    
        try
            % Load data
            [T, G, Ang, wake, rem, sws] = load_mouse_data(DATA_PATH, mouse_name, mouse_by_electrode_brain_region, brain_region);
        catch ME
            ME
            
            try
                % Try one more time after some pause
                pause(5);
                [T, G, Ang, wake, rem, sws] = load_mouse_data(DATA_PATH, mouse_name, mouse_by_electrode_brain_region, brain_region);
            catch ME
                ME
                
                continue;
            end
        end

        for behavioral_state = {'wake', 'rem', 'sws'}
            fprintf('Working on state %s\n', behavioral_state{1});

            try
                tic

                % Analyse data according to behavioral state
                switch behavioral_state{1}
                    case 'wake'
                        period = wake;
                    case 'rem'
                        period = rem;
                    case 'sws'
                        period = sws;
                end

                % Basic extraction of data
                [full_neuron_firing_per_bin, angle_per_temporal_bin] = create_spike_count_and_angles_vector_ver1(period, T, G, Ang);

                %
                filtered_neuron_firing = filter_neuron_firing(full_neuron_firing_per_bin);

                % Reduce data
                full_reduced_data = create_reduced_data(filtered_neuron_firing, P_NEIGHBORS_VEC, NUMBER_OF_REDUCED_DIMENSIONS_VEC);
                % Take the final results and continue processing on them
                reduced_data = full_reduced_data{length(P_NEIGHBORS_VEC) + 1};

                %
                spike_rate_mat_neuron_by_angle = calculate_spike_rate_neuron_by_angle(T, G, Ang, wake);

                %
                estimated_head_direction_angle_per_sample_index = estimate_head_direction(spike_rate_mat_neuron_by_angle, full_neuron_firing_per_bin);

                % Handle missing behavioral entries
                if INCLUDE_UNIDENTIFIED_ANGLES == false && ...
                    strcmp(BEHAVIORAL_STATE, 'wake')
                    reduced_data = reduced_data(~isnan(angle_per_temporal_bin), :);
                    angle_per_temporal_bin = angle_per_temporal_bin(~isnan(angle_per_temporal_bin));
                    estimated_head_direction_angle_per_sample_index = estimated_head_direction_angle_per_sample_index(~isnan(angle_per_temporal_bin));
                end

                % Save results to mouse folder
                save_analysis_results(DATA_PATH, mouse_name, brain_region, behavioral_state{1}, full_neuron_firing_per_bin, full_reduced_data, angle_per_temporal_bin, spike_rate_mat_neuron_by_angle, estimated_head_direction_angle_per_sample_index);

                toc

            catch ME
                ME
            end
        
        end
        
    end
    
end

%% Run with all brain regions together
% TODO: Consider merging the code with the above since it's a complete
% duplicate
for mouse = ALL_MOUSE_NAMES
    mouse_name = mouse{1}{1};
    
    fprintf('Working on mouse %s\n', mouse_name);
    
    try
        % Load data
        [T, G, Ang, wake, rem, sws] = load_mouse_data_all(DATA_PATH, mouse_name);
    catch ME
        ME

        try
            % Try one more time after some pause
            pause(5);
            [T, G, Ang, wake, rem, sws] = load_mouse_data_all(DATA_PATH, mouse_name);
        catch ME
            ME

            continue;
        end
    end

    for behavioral_state = {'wake', 'rem', 'sws'}
        fprintf('Working on state %s\n', behavioral_state{1});

        try
            tic

            % Analyse data according to behavioral state
            switch behavioral_state{1}
                case 'wake'
                    period = wake;
                case 'rem'
                    period = rem;
                case 'sws'
                    period = sws;
            end

            % Basic extraction of data
            [full_neuron_firing_per_bin, angle_per_temporal_bin] = create_spike_count_and_angles_vector_ver1(period, T, G, Ang);

            %
            filtered_neuron_firing = filter_neuron_firing(full_neuron_firing_per_bin);

            % Reduce data
            full_reduced_data = create_reduced_data(filtered_neuron_firing, P_NEIGHBORS_VEC, NUMBER_OF_REDUCED_DIMENSIONS_VEC);
            % Take the final results and continue processing on them
            reduced_data = full_reduced_data{length(P_NEIGHBORS_VEC) + 1};

            %
            spike_rate_mat_neuron_by_angle = calculate_spike_rate_neuron_by_angle(T, G, Ang, wake);

            %
            estimated_head_direction_angle_per_sample_index = estimate_head_direction(spike_rate_mat_neuron_by_angle, full_neuron_firing_per_bin);

            % Handle missing behavioral entries
            if INCLUDE_UNIDENTIFIED_ANGLES == false && ...
                strcmp(BEHAVIORAL_STATE, 'wake')
                reduced_data = reduced_data(~isnan(angle_per_temporal_bin), :);
                angle_per_temporal_bin = angle_per_temporal_bin(~isnan(angle_per_temporal_bin));
                estimated_head_direction_angle_per_sample_index = estimated_head_direction_angle_per_sample_index(~isnan(angle_per_temporal_bin));
            end

            % Save results to mouse folder
            % TODO: '5' here is all brain regions
            save_analysis_results(DATA_PATH, mouse_name, 5, behavioral_state{1}, full_neuron_firing_per_bin, full_reduced_data, angle_per_temporal_bin, spike_rate_mat_neuron_by_angle, estimated_head_direction_angle_per_sample_index);

            toc

        catch ME
            ME
        end
        
    end
    
end