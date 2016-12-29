global TEMPORAL_TIME_BIN;

% Import project-wide constants
constants

MOUSE_BY_DAY_NAME = 'Mouse28-140313';
BRAIN_REGION = 5;
BEHAVIORAL_STATE = 'wake'; % 'wake', 'rem', 'sws'

DATA_PATH = 'E:\or\data\';

INCLUDE_UNIDENTIFIED_ANGLES = true;

%% Load analysis results
[full_reduced_data, full_neuron_firing_per_bin, angle_per_temporal_bin, spike_rate_mat_neuron_by_angle, estimated_head_direction_angle_per_sample_index] = load_analysis_results(DATA_PATH, MOUSE_BY_DAY_NAME, BRAIN_REGION, BEHAVIORAL_STATE);
% TODO: Hard coded 3
reduced_data = full_reduced_data{3};

%% Working
% Polar plot
head_direction_neurons_indices = find_head_direction_neurons(spike_rate_mat_neuron_by_angle);
plot_polar_tuning_curve(spike_rate_mat_neuron_by_angle, head_direction_neurons_indices);

if strcmp(BEHAVIORAL_STATE, 'wake')
    % Decoder visualization
    figure;
    plot([1:length(estimated_head_direction_angle_per_sample_index)] * TEMPORAL_TIME_BIN, estimated_head_direction_angle_per_sample_index, 'r.');
    hold on;
    plot([1:length(angle_per_temporal_bin)] * TEMPORAL_TIME_BIN, angle_per_temporal_bin, 'k.');

    % Plot results of decoder against actual angle
    figure;
    hold on;
    plot(estimated_head_direction_angle_per_sample_index, ...
         angle_per_temporal_bin, 'b.');
    plot([0 2 * pi], [0 2 * pi], 'r-');

    xlim([0 2 * pi]);
    ylim([0 2 * pi]);
end

%%

% Plot the unlabeled reduced data
figure;
plot3(reduced_data(:,2),reduced_data(:,3),reduced_data(:,4),'.');

% Plot the angle on the reduced data
cmap2=hsv(NUMBER_OF_ANGLE_BINS);
% Add black color for 'nan' values
cmap2 = [cmap2; 0 0 0];

% For sleeping behavioral mode (rem and sws) we use the decoded head
% direction rather than actual head direction (which would probably be
% constant).
if strcmp(BEHAVIORAL_STATE, 'wake')
    visualization_angle_per_temporal_bin = angle_per_temporal_bin;
else
    visualization_angle_per_temporal_bin = estimated_head_direction_angle_per_sample_index;
end

index_of_visualization_angle_per_temporal_bin = round(NUMBER_OF_ANGLE_BINS * visualization_angle_per_temporal_bin / ( 2 * pi));
index_of_visualization_angle_per_temporal_bin(index_of_visualization_angle_per_temporal_bin == 0) = NUMBER_OF_ANGLE_BINS;
% Color the missing values in black
index_of_visualization_angle_per_temporal_bin(isnan(index_of_visualization_angle_per_temporal_bin)) = NUMBER_OF_ANGLE_BINS + 1;
% The fourth argument is the dot size
figure;
scatter3(reduced_data(:, 2), reduced_data(:, 3), reduced_data(:, 4), 20, cmap2(index_of_visualization_angle_per_temporal_bin,:), 'fill')

%% Plot colorless and with specific neuron
NEURON_INDEX = 20;

neuron_activity_indices = full_neuron_firing_per_bin(:, NEURON_INDEX) > 4;

% The fourth argument is the dot size
figure;
hold on;
plot(reduced_data(:, 2), reduced_data(:, 3), 'k.');
plot(reduced_data(neuron_activity_indices, 2), reduced_data(neuron_activity_indices, 3), 'r.');