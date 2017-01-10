%% Plot same neurons in different mice
% We create the two different mice separately. Load the first, run this
% code and then load the second and rerun the code.
%% Plot colorless and with specific neuron

EXAMPLE_TYPE = 3;

switch EXAMPLE_TYPE
    case 1
        % M12_120806
        NEURON_INDEX_1 = 35;
        NEURON_INDEX_2 = 20;
        
        MIRROR_DATE = -1;
        
        XLIM = [-0.013 0.02];
        YLIM = [-0.017 0.016];
    case 2
        % M28_140313
        NEURON_INDEX_1 = 18;
        NEURON_INDEX_2 = 44;
        
        MIRROR_DATE = 1;
        
        XLIM = [-0.012 0.015];
        YLIM = [-0.012 0.015];
    case 3
        % M28_140313
        NEURON_INDEX_1 = 7;
        NEURON_INDEX_2 = 44;
        
        MIRROR_DATE = 1;
        
        XLIM = [-0.012 0.015];
        YLIM = [-0.012 0.015];
end

NUMBER_OF_ANGLE_BINS = 40;

CENTER_OF_ANGLE_BINS = [0.5 * (2 * pi) / NUMBER_OF_ANGLE_BINS:...
                        (2 * pi) / NUMBER_OF_ANGLE_BINS:...
                        2 * pi - 0.5 * (2 * pi) / NUMBER_OF_ANGLE_BINS];

neuron_1_activity_indices = full_neuron_firing_per_bin(:, NEURON_INDEX_1) > 1;
neuron_2_activity_indices = full_neuron_firing_per_bin(:, NEURON_INDEX_2) > 3;

% The fourth argument is the dot size
figure;
subplot(2, 1, 1);
hold on;
scatter(reduced_data(:, 2), MIRROR_DATE * reduced_data(:, 3), 1, 'k.');
scatter(reduced_data(neuron_1_activity_indices, 2), MIRROR_DATE * reduced_data(neuron_1_activity_indices, 3), 5, 'r.');
scatter(reduced_data(neuron_2_activity_indices, 2), MIRROR_DATE * reduced_data(neuron_2_activity_indices, 3), 5, 'b.');

axis equal;

xlim(XLIM);
ylim(YLIM);

subplot(2, 1, 2);
neuron_1_spike_rate_by_angle = spike_rate_mat_neuron_by_angle(NEURON_INDEX_1, :);
neuron_2_spike_rate_by_angle = spike_rate_mat_neuron_by_angle(NEURON_INDEX_2, :);

neuron_1_spike_rate_by_angle = neuron_1_spike_rate_by_angle / max(neuron_1_spike_rate_by_angle);
neuron_2_spike_rate_by_angle = neuron_2_spike_rate_by_angle / max(neuron_2_spike_rate_by_angle);

% Connect the last point and the first
polarplot([CENTER_OF_ANGLE_BINS CENTER_OF_ANGLE_BINS(1)], ...
          [neuron_1_spike_rate_by_angle neuron_1_spike_rate_by_angle(1)]);

hold on;

% Connect the last point and the first
polarplot([CENTER_OF_ANGLE_BINS CENTER_OF_ANGLE_BINS(1)], ...
          [neuron_2_spike_rate_by_angle neuron_2_spike_rate_by_angle(1)]);
      
rlim([0 1.2]); 
      
ax = gca;
ax.ThetaTickLabel = [];
ax.RTickLabel = [];