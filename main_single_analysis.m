% Import project-wide constants
constants

MOUSE_BY_DAY_NAME = 'Mouse12-120806';
%MOUSE_BY_DAY_NAME = 'Mouse28-140312';

BEHAVIORAL_STATE = 'wake'; % 'wake', 'rem', 'sws'
% TODO: Notice that if we choose not to include these angles we get
% incorrect results by the decoder because of some delay in the decoded
% angles compared to the actual angle.
INCLUDE_UNIDENTIFIED_ANGLES = true;

SOFTWARE_PATH = 'E:\or\software\';
DATA_PATH = 'E:\or\data\';

addpath([SOFTWARE_PATH 'crcns-hc2-scripts']);
addpath([SOFTWARE_PATH 'custom_scripts']);

%% Load data
% TODO: The following function loads a subset of the recorded neurons.
% Open the function to see the current neurons being loaded.
[T, G, Ang, wake, rem, sws] = load_mouse_data(DATA_PATH, MOUSE_BY_DAY_NAME);

%%
% Analyse data according to behavioral state
switch BEHAVIORAL_STATE
    case 'wake'
        period = wake;
    case 'rem'
        period = rem;
    case 'sws'
        period = sws;
end

%%
[full_neuron_firing_per_bin, reduced_data, angle_per_temporal_bin] = analyze_neuronal_and_behavioral_data(T, G, Ang, period);

%%
spike_rate_mat_neuron_by_angle = calculate_spike_rate_neuron_by_angle(T, G, Ang, wake);

%%
estimated_head_direction_angle_per_sample_index = estimate_head_direction(spike_rate_mat_neuron_by_angle, full_neuron_firing_per_bin);

%% Handle missing behavioral entries
if INCLUDE_UNIDENTIFIED_ANGLES == false && ...
    strcmp(BEHAVIORAL_STATE, 'wake')
    reduced_data = reduced_data(~isnan(angle_per_temporal_bin), :);
    angle_per_temporal_bin = angle_per_temporal_bin(~isnan(angle_per_temporal_bin));
    estimated_head_direction_angle_per_sample_index = estimated_head_direction_angle_per_sample_index(~isnan(angle_per_temporal_bin));
end

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
scatter3(reduced_data(:, 2), reduced_data(:, 3), reduced_data(:, 4), 20, cmap2(index_of_visualization_angle_per_temporal_bin, :), 'fill');

%%
NUMBER_OF_ANGULAR_VELOCITY_BINS = 16;
MAX_ANGULAR_VELOCITY = 0.25;
MIN_ANGULAR_VELOCITY = -0.25;

index_of_visualization_angular_velocity_per_temporal_bin = round(NUMBER_OF_ANGULAR_VELOCITY_BINS * ((angular_velocity - MIN_ANGULAR_VELOCITY) / (MAX_ANGULAR_VELOCITY - MIN_ANGULAR_VELOCITY)));
index_of_visualization_angular_velocity_per_temporal_bin(index_of_visualization_angular_velocity_per_temporal_bin > NUMBER_OF_ANGULAR_VELOCITY_BINS) = NUMBER_OF_ANGULAR_VELOCITY_BINS;
index_of_visualization_angular_velocity_per_temporal_bin(index_of_visualization_angular_velocity_per_temporal_bin < 1) = 1;
index_of_visualization_angular_velocity_per_temporal_bin(isnan(index_of_visualization_angular_velocity_per_temporal_bin)) = NUMBER_OF_ANGULAR_VELOCITY_BINS + 1;

cmap_angular_velocity = jet(NUMBER_OF_ANGULAR_VELOCITY_BINS);
cmap_angular_velocity = [cmap_angular_velocity; [0 0 0]];

figure;
scatter3(reduced_data(:, 2), reduced_data(:, 3), reduced_data(:, 4), 20, cmap_angular_velocity([index_of_visualization_angular_velocity_per_temporal_bin 1], :), 'fill');