function plot_spike_rate_neuron_by_angular_velocity( spike_rate_mat_neuron_by_angular_velocity )
%CALCULATE_SPIKE_RATE_NEURON_BY_ANGULAR_VELOCITY Summary of this function goes here
%   Detailed explanation goes here

    global CENTER_OF_ANGULAR_VELOCITY_BINS;

    number_of_neurons = size(spike_rate_mat_neuron_by_angular_velocity, 1);
    
    number_of_columns = ceil(sqrt(number_of_neurons));
    if number_of_neurons > number_of_columns * (number_of_columns - 1)
        number_of_rows = number_of_columns;
    else
        number_of_rows = number_of_columns - 1;
    end
    
    figure;
    for neuron_index = 1:number_of_neurons
        current_neuron_spike_rate_by_angular_velocity = spike_rate_mat_neuron_by_angular_velocity(neuron_index, :);
        
        subplot(number_of_rows, number_of_columns, neuron_index);
        
        % Connect the last point and the first
        plot(CENTER_OF_ANGULAR_VELOCITY_BINS, current_neuron_spike_rate_by_angular_velocity);        
    end

end

