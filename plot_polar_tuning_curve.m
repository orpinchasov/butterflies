function [ spike_rate_mat_neuron_by_angle ] = plot_polar_tuning_curve( spike_rate_mat_neuron_by_angle, valid_head_direction_neurons, varargin )
    number_of_angle_bins = size(spike_rate_mat_neuron_by_angle, 2);
    
    CENTER_OF_ANGLE_BINS = [0.5 * (2 * pi) / number_of_angle_bins:...
                        (2 * pi) / number_of_angle_bins:...
                        2 * pi - 0.5 * (2 * pi) / number_of_angle_bins];
    
    % TODO: A little bit hard coded
    if length(varargin) == 2
        Map = varargin{1};
        electrode_brain_region  = varargin{2};
    end
    
    number_of_neurons = size(spike_rate_mat_neuron_by_angle, 1);
    
    number_of_columns = ceil(sqrt(number_of_neurons));
    if number_of_neurons > number_of_columns * (number_of_columns - 1)
        number_of_rows = number_of_columns;
    else
        number_of_rows = number_of_columns - 1;
    end
    
    figure;
    for neuron_index = 1:number_of_neurons
        if length(varargin) == 2
            electrode_index = Map(neuron_index, 2);
            brain_region_index = electrode_brain_region(electrode_index);
        else
            brain_region_index = 1;
        end
        
        % Choose color according to brain region
        colors = {'k', 'b', 'r', 'g'};
        line_color = colors{brain_region_index};
        
        current_neuron_spike_rate_by_angle = spike_rate_mat_neuron_by_angle(neuron_index, :);
        
        subplot(number_of_rows, number_of_columns, neuron_index);
        
        % Connect the last point and the first
        polarplot([CENTER_OF_ANGLE_BINS CENTER_OF_ANGLE_BINS(1)], ...
                  [current_neuron_spike_rate_by_angle current_neuron_spike_rate_by_angle(1)]);
        
        angle_of_current_cell = angle(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_ANGLE_BINS)));
        length_of_current_cell = abs(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_ANGLE_BINS))) / sum(abs(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_ANGLE_BINS)));

        hold on;

        r_max = max(current_neuron_spike_rate_by_angle);
        
        if ismember(neuron_index, valid_head_direction_neurons)
            polarplot([1 1] * angle_of_current_cell, [0 r_max], line_color, 'LineWidth', 3);
        else
            polarplot([1 1] * angle_of_current_cell, [0 r_max], line_color);
        end
        
        rlim([0 1.2 * r_max]);        
        
        ax = gca;
        ax.ThetaTickLabel = [];
        ax.RTickLabel = [];
        
        text(pi / 4, 1.2 * r_max, [num2str(r_max, '%10.1f') ' Hz']);
        text(-pi / 4, 1.2 * r_max, ['Rayleigh = ' num2str(length_of_current_cell, '%10.2f')]);
    end
end

