function plot_polar_tuning_curve_by_axes( spike_rate_mat_neuron_by_angle, firing_rate, neurons, start_x, start_y, size_x, size_y )
    number_of_angle_bins = size(spike_rate_mat_neuron_by_angle, 2);

    CENTER_OF_ANGLE_BINS = [0.5 * (2 * pi) / number_of_angle_bins:...
                            (2 * pi) / number_of_angle_bins:...
                            2 * pi - 0.5 * (2 * pi) / number_of_angle_bins];

    spacing = -0.02;
    neuron_index = 1;
    
    first = true;
    
    for col = 2:-1:1
        for row = 4:-1:1
            current_start_x = start_x + (col - 1) * size_x * 2;
            current_start_y = start_y + (row - 1) * size_y;

            current_neuron_actual_firing_rate = spike_rate_mat_neuron_by_angle(neurons(neuron_index), :);
            current_neuron_estimated_firing_rate = firing_rate(neurons(neuron_index), :);
            
            axes('position', [current_start_x + size_x - spacing current_start_y size_x size_y]);

            polarplot([CENTER_OF_ANGLE_BINS CENTER_OF_ANGLE_BINS(1)], ...
                      [current_neuron_actual_firing_rate current_neuron_actual_firing_rate(1)], 'k');

            r_max = max(current_neuron_actual_firing_rate);
            rlim([0 1.2 * r_max]);        

            ax = gca;
            ax.ThetaTickLabel = [];
            ax.RTickLabel = [];

            text(pi / 3.5, 1.1 * r_max, [num2str(r_max, '%10.1f') ' Hz'], 'FontSize', 8);

            neuron_index = neuron_index + 1;
            
            if first
                title('External', 'fontsize', 10);
                
            end
            
            axes('position', [current_start_x - spacing - 0.005 current_start_y size_x size_y]);
            
            
            polarplot([CENTER_OF_ANGLE_BINS CENTER_OF_ANGLE_BINS(1)], ...
                      [current_neuron_estimated_firing_rate current_neuron_estimated_firing_rate(1)], 'r');
            r_max = max(current_neuron_estimated_firing_rate);
            rlim([0 1.2 * r_max]);        

            ax = gca;
            ax.ThetaTickLabel = [];
            ax.RTickLabel = [];

            text(pi / 3.5, 1.1 * r_max, [num2str(r_max, '%10.1f') ' Hz'], 'FontSize', 8);
            
            if first
                title('Internal', 'fontsize', 10);
                first = false;
            end
            
        end
        
        first = true;
        
        % Set the spacing between the two columns of tuning curve pairs
        spacing = 0.03;
    end
end
