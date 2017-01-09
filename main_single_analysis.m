% Import project-wide constants
constants

% Two of our best mice
%MOUSE_BY_DAY_NAME = 'Mouse12-120806';
MOUSE_BY_DAY_NAME = 'Mouse28-140313';

BEHAVIORAL_STATE = 'wake'; % 'wake', 'rem', 'sws'

BRAIN_REGION = 5; % 1 - thalamus, 2 - subiculum, 3 - hippocampus, 4 - prefrontal, 5 - all

% Various configurations for analysis
MOVEMENT_THRESHOLD = 0.08;
NUMBER_OF_ACTIVE_NEURONS_THRESHOLD = 15;

%% Load data

strings = strsplit(MOUSE_BY_DAY_NAME, '-');
[mouse_name, mouse_id] = deal(strings{:});

if BRAIN_REGION == 5
    % All brain regions
    [T, G, Map, Ang, Pos, wake, rem, sws] = load_mouse_data(DATA_PATH, MOUSE_BY_DAY_NAME);
else
    [T, G, Map, Ang, Pos, wake, rem, sws] = load_mouse_data(DATA_PATH, MOUSE_BY_DAY_NAME, mouse_by_electrode_brain_region, BRAIN_REGION);
end

%% Basic extraction of data

% Analyse data according to behavioral state
switch BEHAVIORAL_STATE
    case 'wake'
        period = wake;
    case 'rem'
        period = rem;
    case 'sws'
        period = sws;
end

[full_neuron_firing_per_bin, angle_per_temporal_bin, position_per_temporal_bin] = create_spike_count_and_angles_vector_ver1(period, T, G, Ang, Pos);

% Filter the neurons firing
filtered_full_neuron_firing = filter_neuron_firing(full_neuron_firing_per_bin);


%%% Run either one of the following three blocks to get different types of
%%% filtering:
%%% 1) According to no movement frames (useful to remove noise from sleep
%%% phases)
%%% 2) According to a minimum number of total active neurons in a specific
%%% sample (useful again to remove quiet frames which contribute to total
%%% noise levels)
%%% 3) No filtration
%% Filter according to no movement frames
% Calculate by testing the diff in position between every two frames and
% remove frames which contain significant motion.
% It was thought to improve the results of the REM which might include
% frames in which the mouse has awakened and moved.

% Calculate amount of movement
position_diff = diff(position_per_temporal_bin);
% Take the values of a single LED (there are 2 on the mouse)
amount_of_movement = sqrt(position_diff(:, 1).^2 + position_diff(:, 2).^2);
% Add the missing value at the end
amount_of_movement = [amount_of_movement; 0];

filter_mask = amount_of_movement < MOVEMENT_THRESHOLD;

%% Filter according to frames which contain a minimum number of active neurons
% (filter out quiet frames).

% Count number of active neurons per sample
number_of_active_neurons = sum(filtered_full_neuron_firing, 2);

filter_mask = number_of_active_neurons > NUMBER_OF_ACTIVE_NEURONS_THRESHOLD;

%% Perform no filtering of the frames

filter_mask = true(length(full_neuron_firing_per_bin), 1);

%% Truncate data according to actual frames being used

filtered_neuron_firing = filtered_full_neuron_firing(filter_mask, :);
filtered_angle_per_temporal_bin = angle_per_temporal_bin(filter_mask);
filtered_position_per_temporal_bin = position_per_temporal_bin(filter_mask);

%% Plotting of basic data extracted
% TODO: Complete if necessary

%% Reduce data
% Original P values
P_NEIGHBORS_VEC = [0.025 / 30 0.025];
NUMBER_OF_REDUCED_DIMENSIONS_VEC = [10 10];
    
full_reduced_data = create_reduced_data(filtered_neuron_firing, P_NEIGHBORS_VEC, NUMBER_OF_REDUCED_DIMENSIONS_VEC);
% Take the final results and continue processing on them
reduced_data = full_reduced_data{length(P_NEIGHBORS_VEC) + 1};

spike_rate_mat_neuron_by_angle = calculate_spike_rate_neuron_by_angle(T, G, Ang, wake);

estimated_head_direction_angle_per_sample_index = estimate_head_direction(spike_rate_mat_neuron_by_angle, full_neuron_firing_per_bin);

% Filter the estimated head direction
estimated_head_direction_angle_per_sample_index = estimated_head_direction_angle_per_sample_index(filter_mask);

% Handle missing behavioral entries
if INCLUDE_UNIDENTIFIED_ANGLES == false && ...
    strcmp(BEHAVIORAL_STATE, 'wake')
    reduced_data = reduced_data(~isnan(filtered_angle_per_temporal_bin), :);
    filtered_angle_per_temporal_bin = filtered_angle_per_temporal_bin(~isnan(filtered_angle_per_temporal_bin));
    estimated_head_direction_angle_per_sample_index = estimated_head_direction_angle_per_sample_index(~isnan(filtered_angle_per_temporal_bin));
end

%% Working
% Polar plot
mouse = mouse_by_electrode_brain_region(mouse_name);
electrode_brain_region = mouse(mouse_id);
head_direction_neurons_indices = find_head_direction_neurons(spike_rate_mat_neuron_by_angle);
plot_polar_tuning_curve(spike_rate_mat_neuron_by_angle, head_direction_neurons_indices, Map, electrode_brain_region);

%
if strcmp(BEHAVIORAL_STATE, 'wake')
    % Decoder visualization
    figure;
    plot([1:length(estimated_head_direction_angle_per_sample_index)] * TEMPORAL_TIME_BIN, estimated_head_direction_angle_per_sample_index, 'r.');
    hold on;
    plot([1:length(filtered_angle_per_temporal_bin)] * TEMPORAL_TIME_BIN, filtered_angle_per_temporal_bin, 'k.');

    % Plot results of decoder against actual angle
    figure;
    hold on;
    plot(estimated_head_direction_angle_per_sample_index, ...
         filtered_angle_per_temporal_bin, 'b.');
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
    visualization_angle_per_temporal_bin = filtered_angle_per_temporal_bin;
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
figure;
scatter3(reduced_data(:, 5), reduced_data(:, 6), reduced_data(:, 7), 20, cmap2(index_of_visualization_angle_per_temporal_bin, :), 'fill');

figure;
scatter(reduced_data(:, 2), reduced_data(:, 3), 5, cmap2(index_of_visualization_angle_per_temporal_bin, :), 'fill');

%% Plot the rapid firing neurons to check their positions on the scatter plot
significant_neurons = [25 43 44 46 49 54 55 59 60];

%neuron_activity_indices = filtered_neuron_firing(:, 3) > 0;

%figure;
%hold on;
%scatter(reduced_data(:, 2), reduced_data(:, 3), 5, '.k');
%scatter(reduced_data(neuron_activity_indices, 2), reduced_data(neuron_activity_indices, 3), 20, '.r');

for i = 1:length(full_neuron_firing_per_bin)
    neuron_activity_indices = filtered_neuron_firing(:, i) > 0;
    
    figure;
    hold on;
    scatter(reduced_data(:, 2), reduced_data(:, 3), 5, '.k');
    scatter(reduced_data(neuron_activity_indices, 2), reduced_data(neuron_activity_indices, 3), 20, '.r');
    
    saveas(gca, ['E:\or\results\fig4_ver3\neurons_position_on_ring\' num2str(i) '.jpg'], 'jpg');
end

%% Choose 4 points on the reduced data plot and get the indices of all points in the polygon created by these points
figure;
scatter(reduced_data(:, 2), reduced_data(:, 3), 5, cmap2(index_of_visualization_angle_per_temporal_bin, :), 'fill');
[x y] = ginput(4);

%%
points_in_polygon = inpolygon(reduced_data(:, 2), reduced_data(:, 3), x, y);

figure;
scatter(reduced_data(points_in_polygon, 2), reduced_data(points_in_polygon, 3), '.');

figure;
hold on;
plot(amount_of_movement);

%% Plot scattering of REM events overlayed with total number of active neurons per time frame.
cmap_number_of_active_neurons = hsv(max(number_of_active_neurons));

figure; scatter(reduced_data(:, 2), reduced_data(:, 3), 20, cmap_number_of_active_neurons(number_of_active_neurons(number_of_active_neurons > NUMBER_OF_ACTIVE_NEURONS_THRESHOLD), :), '.'); colorbar;

figure; scatter3(reduced_data(:, 2), reduced_data(:, 3), number_of_active_neurons(number_of_active_neurons > NUMBER_OF_ACTIVE_NEURONS_THRESHOLD), 20, '.');

%% Calculate angular velocity
NUMBER_OF_ANGULAR_VELOCITY_BINS = 16;
MAX_ANGULAR_VELOCITY = 0.25;
MIN_ANGULAR_VELOCITY = -0.25;

index_of_visualization_angular_velocity_per_temporal_bin = round(NUMBER_OF_ANGULAR_VELOCITY_BINS * ((angular_velocity - MIN_ANGULAR_VELOCITY) / (MAX_ANGULAR_VELOCITY - MIN_ANGULAR_VELOCITY)));
index_of_visualization_angular_velocity_per_temporal_bin(index_of_visualization_angular_velocity_per_temporal_bin > NUMBER_OF_ANGULAR_VELOCITY_BINS) = NUMBER_OF_ANGULAR_VELOCITY_BINS;
index_of_visualization_angular_velocity_per_temporal_bin(index_of_visualization_angular_velocity_per_temporal_bin < 1) = 1;
index_of_visualization_angular_velocity_per_temporal_bin(isnan(index_of_visualization_angular_velocity_per_temporal_bin)) = NUMBER_OF_ANGULAR_VELOCITY_BINS + 1;

%% Run the following before:
% [spike_rate_mat_neuron_by_angular_velocity index_of_visualization_angular_velocity_per_temporal_bin] = calculate_spike_rate_neuron_by_angular_velocity(full_neuron_firing_per_bin, angle_per_temporal_bin);
cmap_angular_velocity = jet(NUMBER_OF_ANGULAR_VELOCITY_BINS / 2);
cmap_angular_velocity = [cmap_angular_velocity; [0 0 0]];

index_of_visualization_abs_angular_velocity_per_temporal_bin = ceil(abs(index_of_visualization_angular_velocity_per_temporal_bin - (NUMBER_OF_ANGULAR_VELOCITY_BINS / 2 + 0.5)));
index_of_visualization_abs_angular_velocity_per_temporal_bin(index_of_visualization_angular_velocity_per_temporal_bin == NUMBER_OF_ANGULAR_VELOCITY_BINS + 1) = NUMBER_OF_ANGULAR_VELOCITY_BINS / 2 + 1;

figure;
scatter3(reduced_data(:, 2), reduced_data(:, 3), reduced_data(:, 4), 20, cmap_angular_velocity(index_of_visualization_abs_angular_velocity_per_temporal_bin, :), 'fill');

%% Clustering tests
NUMBER_OF_CLUSTERS = 8;

cmap_clustering = jet(NUMBER_OF_CLUSTERS);

clustering_results = k_means_clustering(reduced_data(:, 2:7), NUMBER_OF_CLUSTERS, 1);

figure;
%scatter3(reduced_data(:, 2), reduced_data(:, 3), reduced_data(:, 4), 20, cmap_clustering(clustering_results, :), 'fill');
scatter(reduced_data(:, 2), reduced_data(:, 3), 20, cmap_clustering(clustering_results, :), 'fill');

axis equal;
box;

xlim([-0.012 0.015]);
ylim([-0.013 0.014]);

xlabel('Comp. 1');
ylabel('Comp. 2');

%% Spectral clustering tests (run either this or the previous part)
NUMBER_OF_CLUSTERS = 8;
NUMBER_OF_SAMPLES = 20000;

cmap_clustering = jet(NUMBER_OF_CLUSTERS);

%spectral_clustering_data = reduced_data(1:NUMBER_OF_SAMPLES, 2:4);
%spectral_clustering_distances = squareform(pdist(spectral_clustering_data));
%similarity_spectral_clustering_distances = exp(-spectral_clustering_distances ./ repmat(std(spectral_clustering_distances), [NUMBER_OF_SAMPLES 1]));

M = reduced_data(1:NUMBER_OF_SAMPLES, 2:7);
W = squareform(pdist(M));
sigma = 0.05;
similarity_spectral_clustering_distances = exp(-W.^2 ./ (2*sigma^2));

%figure;
%imagesc(similarity_spectral_clustering_distances);
%colorbar;
%%

raw_clustering_results = SpectralClustering(similarity_spectral_clustering_distances, NUMBER_OF_CLUSTERS, 3);

clustering_results = log2(bi2de(raw_clustering_results)) + 1;

figure;
%scatter3(reduced_data(1:NUMBER_OF_SAMPLES, 2), reduced_data(1:NUMBER_OF_SAMPLES, 3), reduced_data(1:NUMBER_OF_SAMPLES, 4), 20, cmap_clustering(mapped_clustering_results, :), 'fill');
scatter(reduced_data(1:NUMBER_OF_SAMPLES, 2), reduced_data(1:NUMBER_OF_SAMPLES, 3), 20, cmap_clustering(clustering_results, :), 'fill');

%% Load clustering results
% For example, from the file
% "E:\or\data\Mouse28-140313\output\all\wake\spectral_clustered_2d_data_m28_140313_wake_all.mat"
LABELS_FILENAME = 'E:\or\data\Mouse28-140313\output\all\wake\spectral_clustered_2d_data_m28_140313_wake_all.mat';

labels = load('E:\or\data\Mouse28-140313\output\all\wake\spectral_clustered_2d_data_m28_140313_wake_all.mat');
clustering_results = labels.labels' + 1;

%% Test clustering using silhouette
eva = evalclusters(reduced_data, 'kmeans', 'Silhouette', 'KList', 1:8);

%% Plot clustering histograms versus actual head direction
figure;
for figure_index = 1:NUMBER_OF_CLUSTERS
    subplot(NUMBER_OF_CLUSTERS, 1, figure_index);
    hist(filtered_angle_per_temporal_bin(clustering_results == figure_index), CENTER_OF_ANGLE_BINS);
    %hist(angle_per_temporal_bin(clustering_results == chosen_shuffle(figure_index)), CENTER_OF_ANGLE_BINS);
end

%% Transition matrix
transition_index_vec = clustering_results(1:end - 1) + (clustering_results(2:end) - 1) * NUMBER_OF_CLUSTERS;
[transition_index_count, ~] = histcounts(transition_index_vec, [0.5:1:NUMBER_OF_CLUSTERS^2 + 0.5]);
transition_index_mat = reshape(transition_index_count, [NUMBER_OF_CLUSTERS NUMBER_OF_CLUSTERS])';

transition_mat = transition_index_mat ./ repmat(sum(transition_index_mat, 2), [1 NUMBER_OF_CLUSTERS]);

figure; imagesc(transition_mat); colormap jet;

%% Run ordering code to get correct shuffling of the matrix
Ordering_cyclical_Ver000

%% Plot for each neuron the firing rate per cluster in sample

number_of_neurons = size(full_neuron_firing_per_bin, 2);

neuron_by_cluster_spike_count = zeros(NUMBER_OF_CLUSTERS, number_of_neurons);

% Count the number of frames of each cluster
frames_per_cluster_count = histcounts(clustering_results,  0.5:1:NUMBER_OF_CLUSTERS + 0.5);

for cluster_index = 1:NUMBER_OF_CLUSTERS
    cluster_frames_indices = find(clustering_results == cluster_index);
    
    neuron_by_cluster_spike_count(cluster_index, :) = sum(full_neuron_firing_per_bin(cluster_frames_indices, :), 1);
end

neuron_firing_rate = (neuron_by_cluster_spike_count ./ repmat(frames_per_cluster_count', [1 number_of_neurons])) * (BEHAVIORAL_SAMPLE_RATE / BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN);

ordered_neuron_firing_rate = neuron_firing_rate(chosen_shuffle, :);

%% Polar plot tuning curves created by clustering (all neurons)
CENTER_OF_CLUSTERING_ANGLE_BINS = 0.5 * (2 * pi) / NUMBER_OF_CLUSTERS:...
                                  (2 * pi) / NUMBER_OF_CLUSTERS:...
                                  2 * pi - 0.5 * (2 * pi) / NUMBER_OF_CLUSTERS;

number_of_columns = ceil(sqrt(number_of_neurons));
if number_of_neurons > number_of_columns * (number_of_columns - 1)
    number_of_rows = number_of_columns;
else
    number_of_rows = number_of_columns - 1;
end

figure;
for neuron_index = 1:number_of_neurons
    current_neuron_spike_rate_by_angle = ordered_neuron_firing_rate(:, neuron_index)';

    subplot(number_of_rows, number_of_columns, neuron_index);

    % Connect the last point and the first
    polarplot([CENTER_OF_CLUSTERING_ANGLE_BINS CENTER_OF_CLUSTERING_ANGLE_BINS(1)] + 0.5 * pi, ...
              [current_neuron_spike_rate_by_angle current_neuron_spike_rate_by_angle(1)]);

    angle_of_current_cell = angle(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_CLUSTERING_ANGLE_BINS)));
    length_of_current_cell = abs(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_CLUSTERING_ANGLE_BINS))) / sum(abs(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_CLUSTERING_ANGLE_BINS)));

    hold on;

    r_max = max(current_neuron_spike_rate_by_angle);

    %if ismember(neuron_index, valid_head_direction_neurons)
    %    polarplot([1 1] * angle_of_current_cell, [0 r_max], 'r', 'LineWidth', 2);
    %else
        polarplot([1 1] * (angle_of_current_cell + 0.5 * pi), [0 r_max], 'k');
    %end

    rlim([0 1.2 * r_max]);        

    ax = gca;
    ax.ThetaTickLabel = [];
    ax.RTickLabel = [];

    text(pi / 4, 1.2 * r_max, [num2str(r_max, '%10.1f') ' Hz']);
    text(-pi / 4, 1.2 * r_max, ['Rayleigh = ' num2str(length_of_current_cell, '%10.2f')]);
end

%% Polar plot tuning curves created by clustering (only head direction)
CENTER_OF_CLUSTERING_ANGLE_BINS = 0.5 * (2 * pi) / NUMBER_OF_CLUSTERS:...
                                  (2 * pi) / NUMBER_OF_CLUSTERS:...
                                  2 * pi - 0.5 * (2 * pi) / NUMBER_OF_CLUSTERS;

number_of_head_direction_neurons = length(head_direction_neurons_indices);
                              
number_of_columns = ceil(sqrt(number_of_head_direction_neurons));
if number_of_head_direction_neurons > number_of_columns * (number_of_columns - 1)
    number_of_rows = number_of_columns;
else
    number_of_rows = number_of_columns - 1;
end

figure;
for neuron_index = 1:number_of_head_direction_neurons
    current_neuron_spike_rate_by_angle = ordered_neuron_firing_rate(:, head_direction_neurons_indices(neuron_index))';

    subplot(number_of_rows, number_of_columns, neuron_index);

    % Connect the last point and the first
    polarplot([CENTER_OF_CLUSTERING_ANGLE_BINS CENTER_OF_CLUSTERING_ANGLE_BINS(1)] + 0.5 * pi, ...
              [current_neuron_spike_rate_by_angle current_neuron_spike_rate_by_angle(1)]);

    angle_of_current_cell = angle(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_CLUSTERING_ANGLE_BINS)));
    length_of_current_cell = abs(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_CLUSTERING_ANGLE_BINS))) / sum(abs(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_CLUSTERING_ANGLE_BINS)));

    hold on;

    r_max = max(current_neuron_spike_rate_by_angle);

    %if ismember(neuron_index, valid_head_direction_neurons)
    %    polarplot([1 1] * angle_of_current_cell, [0 r_max], 'r', 'LineWidth', 2);
    %else
        polarplot([1 1] * (angle_of_current_cell + 0.5 * pi), [0 r_max], 'k');
    %end

    rlim([0 1.2 * r_max]);        

    ax = gca;
    ax.ThetaTickLabel = [];
    ax.RTickLabel = [];

    text(pi / 4, 1.2 * r_max, [num2str(r_max, '%10.1f') ' Hz']);
    text(-pi / 4, 1.2 * r_max, ['Rayleigh = ' num2str(length_of_current_cell, '%10.2f')]);
end

%% Plot scattering of clustering tuning curve versus actual tuning curve
SLOPE_MULTIPLIER = -1; % Either 1 or -1 to turn slope
ACTUAL_VERSUS_CLUSTERING_SHIFT = 0.5 * pi;

neuron_actual_preferred_angle = zeros(NUMBER_OF_ANGLE_BINS, 1);
neuron_clustering_preferred_angle = zeros(NUMBER_OF_CLUSTERS, 1);

for neuron_index = 1:number_of_neurons
    current_neuron_spike_rate_by_angle = spike_rate_mat_neuron_by_angle(neuron_index, :);

    neuron_actual_preferred_angle(neuron_index) = angle(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_ANGLE_BINS)));
end

for neuron_index = 1:number_of_neurons
    current_neuron_spike_rate_by_angle = ordered_neuron_firing_rate(:, neuron_index)';

    neuron_clustering_preferred_angle(neuron_index) = angle(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_CLUSTERING_ANGLE_BINS)));
end

% Used to get the slope to be positive (bottom left of figure to upper
% right)
corrected_neuron_clustering_preferred_angle = mod(SLOPE_MULTIPLIER * (neuron_clustering_preferred_angle - ACTUAL_VERSUS_CLUSTERING_SHIFT), 2 * pi);

head_direction_neurons_indices = find_head_direction_neurons(spike_rate_mat_neuron_by_angle);

figure;
hold on;
box on;

% Original code to plot uncorrected scattering
%plot([0 ACTUAL_VERSUS_CLUSTERING_SHIFT] - pi, SLOPE_MULTIPLIER * ([2 * pi - ACTUAL_VERSUS_CLUSTERING_SHIFT 2 * pi] - pi), 'r-');
%plot([ACTUAL_VERSUS_CLUSTERING_SHIFT 2 * pi] - pi, SLOPE_MULTIPLIER * ([0 2 * pi - ACTUAL_VERSUS_CLUSTERING_SHIFT] - pi), 'r-');
%scatter(neuron_actual_preferred_angle, neuron_clustering_preferred_angle, 'b');
%scatter(neuron_actual_preferred_angle(head_direction_neurons_indices), neuron_clustering_preferred_angle(head_direction_neurons_indices), 'fill', 'b');

plot([0 2 * pi], [0 2 * pi], 'r-');
scatter(mod(neuron_actual_preferred_angle, 2 * pi), corrected_neuron_clustering_preferred_angle, 'b');
scatter(mod(neuron_actual_preferred_angle(head_direction_neurons_indices), 2 * pi), corrected_neuron_clustering_preferred_angle(head_direction_neurons_indices), 'fill', 'b');

axis equal;

xlim([0 2 * pi]);
ylim([0 2 * pi]);

xlabel('Preferred head direction (rad)');
ylabel('Reconstructed preferred head direction (rad)');

%% Plot same as above for Reighly vector length

neuron_actual_vector_length = zeros(NUMBER_OF_ANGLE_BINS, 1);
neuron_clustering_vector_length = zeros(NUMBER_OF_CLUSTERS, 1);

for neuron_index = 1:number_of_neurons
    current_neuron_spike_rate_by_angle = spike_rate_mat_neuron_by_angle(neuron_index, :);

    neuron_actual_vector_length(neuron_index) = abs(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_ANGLE_BINS))) ./ sum(abs(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_ANGLE_BINS)));
end

for neuron_index = 1:number_of_neurons
    current_neuron_spike_rate_by_angle = ordered_neuron_firing_rate(:, neuron_index)';

    neuron_clustering_vector_length(neuron_index) = abs(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_CLUSTERING_ANGLE_BINS))) ./ sum(abs(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_CLUSTERING_ANGLE_BINS)));
end

figure;
hold on;
box on;

plot([0 2 * pi], [0 2 * pi], 'r-');
scatter(neuron_actual_vector_length, neuron_clustering_vector_length, 'b');
scatter(neuron_actual_vector_length(head_direction_neurons_indices), neuron_clustering_vector_length(head_direction_neurons_indices), 'fill', 'b');

axis equal;

xlim([0 1]);
ylim([0 1]);

xlabel('Head direction tuning vector length');
ylabel('Reconstructed tuning vector length');

%% Plot the estimated polar plots
mouse = 'Mouse28';
mouse = mouse_by_electrode_brain_region(mouse);
id = mouse('140313');

firing_rate = create_firing_rate_matrix(full_neuron_firing_per_bin, smoothed_estimated_angle_by_clustering);
plot_polar_tuning_curve(firing_rate, head_direction_neurons_indices, Map, id);
plot_polar_tuning_curve(spike_rate_mat_neuron_by_angle, head_direction_neurons_indices, Map, id);

%% Plot actual versus estimated head direction polar plot
number_of_angle_bins = size(spike_rate_mat_neuron_by_angle, 2);

% TODO: This should be separated in case the number of bins of the actual
% and the estimated differs
CENTER_OF_ANGLE_BINS = [0.5 * (2 * pi) / number_of_angle_bins:...
                        (2 * pi) / number_of_angle_bins:...
                        2 * pi - 0.5 * (2 * pi) / number_of_angle_bins];

% two directions:
% 45 47
% 
% one direction:
% 43 46 55 56 61
% 
% no direction:
% 62
EXAMPLE_NEURONS = [43 46 55 56 61 45 47 62];

figure;
hold on;
for neuron_index = 1:length(EXAMPLE_NEURONS);
    current_neuron_actual_firing_rate = spike_rate_mat_neuron_by_angle(EXAMPLE_NEURONS(neuron_index), :);
    current_neuron_estimated_firing_rate = firing_rate(EXAMPLE_NEURONS(neuron_index), :);
    
    actual_angle_of_current_neuron = angle(sum(current_neuron_actual_firing_rate .* exp(1i * CENTER_OF_ANGLE_BINS)));
    estimated_angle_of_current_neuron = angle(sum(current_neuron_estimated_firing_rate .* exp(1i * CENTER_OF_ANGLE_BINS)));

    actual_length_of_current_neuron = abs(sum(current_neuron_actual_firing_rate .* exp(1i * CENTER_OF_ANGLE_BINS))) / sum(abs(current_neuron_actual_firing_rate .* exp(1i * CENTER_OF_ANGLE_BINS)));
    estimated_length_of_current_neuron = abs(sum(current_neuron_estimated_firing_rate .* exp(1i * CENTER_OF_ANGLE_BINS))) / sum(abs(current_neuron_estimated_firing_rate .* exp(1i * CENTER_OF_ANGLE_BINS)));
    
    subplot(4, 4, 2 * neuron_index - 1);
    
    polarplot([CENTER_OF_ANGLE_BINS CENTER_OF_ANGLE_BINS(1)], ...
              [current_neuron_actual_firing_rate current_neuron_actual_firing_rate(1)]);
          
    r_max = max(current_neuron_actual_firing_rate);
    rlim([0 1.2 * r_max]);        

    ax = gca;
    ax.ThetaTickLabel = [];
    ax.RTickLabel = [];

    text(pi / 4, 1.2 * r_max, [num2str(r_max, '%10.1f') ' Hz']);
    % TODO: The calculation of the length is incorrect!
    %text(-pi / 4, 1.2 * r_max, ['Rayleigh = ' num2str(length_of_current_cell, '%10.2f')]);

    subplot(4, 4, 2 * neuron_index);
    polarplot([CENTER_OF_ANGLE_BINS CENTER_OF_ANGLE_BINS(1)], ...
              [current_neuron_estimated_firing_rate current_neuron_estimated_firing_rate(1)], 'r');
    r_max = max(current_neuron_estimated_firing_rate);
    rlim([0 1.2 * r_max]);        

    ax = gca;
    ax.ThetaTickLabel = [];
    ax.RTickLabel = [];

    text(pi / 4, 1.2 * r_max, [num2str(r_max, '%10.1f') ' Hz']);
    %text(-pi / 4, 1.2 * r_max, ['Rayleigh = ' num2str(length_of_current_cell, '%10.2f')]);
end

%% Calculate correlation over actual and inferred tuning curve
number_of_angle_bins = size(spike_rate_mat_neuron_by_angle, 2);

correlations = ones(1, number_of_neurons);

% TODO: This should be separated in case the number of bins of the actual
% and the estimated differs
CENTER_OF_ANGLE_BINS = [0.5 * (2 * pi) / number_of_angle_bins:...
                        (2 * pi) / number_of_angle_bins:...
                        2 * pi - 0.5 * (2 * pi) / number_of_angle_bins];
                    
CENTER_OF_HISTOGRAM_BINS = -0.9:0.2:0.9;

for neuron_index = 1:number_of_neurons;
    current_neuron_actual_firing_rate = spike_rate_mat_neuron_by_angle(neuron_index, :);
    current_neuron_estimated_firing_rate = firing_rate(neuron_index, :);
    
    correlations(neuron_index) = corr(current_neuron_actual_firing_rate', current_neuron_estimated_firing_rate');
end

figure;
hist(correlations, 8);


hist_mat = [hist(correlations(head_direction_neurons_indices), CENTER_OF_HISTOGRAM_BINS); ...
            hist(correlations(~ismember(1:number_of_neurons, head_direction_neurons_indices)), CENTER_OF_HISTOGRAM_BINS)];

figure;

bar(CENTER_OF_HISTOGRAM_BINS, hist_mat', 'stacked');

xlabel('Correlation');
ylabel('Count');

%% Trajectory of actual head movement versus clustered movement
ordered_clustering_results = zeros(size(clustering_results));

for cluster_index = 1:NUMBER_OF_CLUSTERS
    cluster_indices = find(clustering_results == chosen_shuffle(cluster_index));
    % TODO: The 9 here is temporary and is needed when the slope multiplier
    % is -1
    ordered_clustering_results(cluster_indices) = 9 - cluster_index;
end

estimated_angle_by_clustering = mod(CENTER_OF_CLUSTERING_ANGLE_BINS(ordered_clustering_results) + ACTUAL_VERSUS_CLUSTERING_SHIFT, 2 * pi);

SmoothingEstimatedHeadDirection_Ver0

figure;
scatter(filtered_angle_per_temporal_bin, smoothed_estimated_angle_by_clustering, 'k.');
xlim([0 2 * pi]);
ylim([0 2 * pi]);
axis square;

%figure;
%hold on;

%plot(angle_per_temporal_bin, 'k.');
%plot(estimated_angle_by_clustering, 'r.');
%plot(2 * pi - smoothed_estimated_angle_by_clustering, 'r.');
%scatter(angle_per_temporal_bin, estimated_angle_by_clustering, '.');

%% PCA over firing rate
[coeff,score,latent,tsquared,explained,mu] = pca(ordered_neuron_firing_rate);
figure; scatter(score(:, 1), score(:, 2), 40, cmap_clusters, 'fill'); axis equal;

%% PCA over transition probability
ordered_transition_mat = transition_mat(chosen_shuffle, chosen_shuffle);
symmetric_ordered_transition_mat = 0.5 * (ordered_transition_mat + ordered_transition_mat');

%% Plot averaged cluster data (in order to compare with the transition graph)
cmap_clusters = hsv(NUMBER_OF_CLUSTERS);

average_cluster_point = ones(NUMBER_OF_CLUSTERS, 2);
for cluster_index = 1:NUMBER_OF_CLUSTERS
    cluster_indices = find(clustering_results == chosen_shuffle(cluster_index));
    average_cluster_point(cluster_index, :) = mean(reduced_data(cluster_indices, 2:3));
end

figure; scatter(average_cluster_point(:, 1), average_cluster_point(:, 2), 300, cmap_clusters, 'fill');

axis equal;
box;

xlim([-0.012 0.015]);
ylim([-0.013 0.014]);

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