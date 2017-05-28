%%
cd('C:\Users\orp\Downloads\matlab_examples');
load_javaplex;
cd('d:\dev\alon_data\code\custom_scripts');

%%
% Import project-wide constants
constants

% Two of our best mice
%MOUSE_BY_DAY_NAME = 'Mouse12-120806';
MOUSE_BY_DAY_NAME = 'Mouse28-140313';

BEHAVIORAL_STATE = 'wake'; % 'wake', 'rem', 'sws'

BRAIN_REGION = 5; % 1 - thalamus, 2 - subiculum, 3 - hippocampus, 4 - prefrontal, 5 - all

% Various configurations for analysis
MOVEMENT_THRESHOLD = 0.08;
% For 'wake'
NUMBER_OF_ACTIVE_NEURONS_THRESHOLD = 15;
% For 'wake' thalamus only
%NUMBER_OF_ACTIVE_NEURONS_THRESHOLD = 0;
% For 'rem'
%NUMBER_OF_ACTIVE_NEURONS_THRESHOLD = 15;

NUMBER_OF_CLUSTERS = 8;
CLUSTERING_DIMENSIONS = 2:7;

% Wake all filtered
ACTUAL_VERSUS_CLUSTERING_SHIFT = 2.75 * pi;

% Wake all unfiltered
%ACTUAL_VERSUS_CLUSTERING_SHIFT = 1.4 * pi;

% REM all unfiltered
%ACTUAL_VERSUS_CLUSTERING_SHIFT = 1.1 * pi;

% REM all filtered
%ACTUAL_VERSUS_CLUSTERING_SHIFT = 0.4 * pi;

% Wake
MIRROR_ORDERING = false;

% REM
%MIRROR_ORDERING = true;

% Derived constants
CENTER_OF_CLUSTERING_ANGLE_BINS = 0.5 * (2 * pi) / NUMBER_OF_CLUSTERS:...
                                  (2 * pi) / NUMBER_OF_CLUSTERS:...
                                  2 * pi - 0.5 * (2 * pi) / NUMBER_OF_CLUSTERS;


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
number_of_neurons = size(full_neuron_firing_per_bin, 2);

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

filtered_full_neuron_firing_per_bin = full_neuron_firing_per_bin(filter_mask, :);
filtered_neuron_firing = filtered_full_neuron_firing(filter_mask, :);
filtered_angle_per_temporal_bin = angle_per_temporal_bin(filter_mask)';
filtered_position_per_temporal_bin = position_per_temporal_bin(filter_mask)';

%% Plotting of basic data extracted
% TODO: Complete if necessary

%% Reduce data
% Original P values
P_NEIGHBORS_VEC = [0.075 / 30 0.075];
NUMBER_OF_REDUCED_DIMENSIONS_VEC = [10 10];
    
full_reduced_data = create_reduced_data(filtered_neuron_firing, P_NEIGHBORS_VEC, NUMBER_OF_REDUCED_DIMENSIONS_VEC);
% Take the final results and continue processing on them
reduced_data = full_reduced_data{length(P_NEIGHBORS_VEC) + 1};

spike_rate_mat_neuron_by_angle = calculate_spike_rate_neuron_by_angle(T, G, Ang, wake);

% Use decoder to estimate head direction
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

head_direction_neurons_indices = find_head_direction_neurons(spike_rate_mat_neuron_by_angle);

%% Visualize decoder performances
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

%% Working
% Polar plot
mouse = mouse_by_electrode_brain_region(mouse_name);
electrode_brain_region = mouse(mouse_id);
plot_polar_tuning_curve(spike_rate_mat_neuron_by_angle, head_direction_neurons_indices, Map, electrode_brain_region);

%%
% Plot the unlabeled reduced data
figure;
plot3(reduced_data(:,2),reduced_data(:,3),reduced_data(:,4),'.');

% Plot the angle on the reduced data
cmap2 = hsv(NUMBER_OF_ANGLE_BINS);
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
scatter(reduced_data(:, 2), reduced_data(:, 3), 5, cmap2(index_of_visualization_angle_per_temporal_bin, :), 'fill');


%%% Choose one of the clustering methods from the following
%% K-Means clustering
clustering_labels = k_means_clustering(reduced_data(:, CLUSTERING_DIMENSIONS), NUMBER_OF_CLUSTERS, 1);

%% Spectral clustering

M = reduced_data(:, CLUSTERING_DIMENSIONS);
W = squareform(pdist(M));
sigma = 0.05;
similarity_spectral_clustering_distances = exp(-W.^2 ./ (2*sigma^2));

raw_clustering_results = SpectralClustering(similarity_spectral_clustering_distances, NUMBER_OF_CLUSTERS, 3);

clustering_labels = log2(bi2de(raw_clustering_results)) + 1;

%% Load clustering results from Python
% For example, from the file
% "E:\or\data\Mouse28-140313\output\all\wake\spectral_clustered_2d_data_m28_140313_wake_all.mat"
LABELS_FILENAME = 'E:\or\data\Mouse28-140313\output\all\wake\spectral_clustered_2d_data_m28_140313_wake_all.mat';

labels = load('E:\or\data\Mouse28-140313\output\all\wake\spectral_clustered_2d_data_m28_140313_wake_all.mat');
clustering_labels = labels.labels' + 1;

%% Clustering visualization
cmap_clustering = jet(NUMBER_OF_CLUSTERS);

figure;
scatter(reduced_data(:, 2), reduced_data(:, 3), 20, cmap_clustering(clustering_labels, :), 'fill');

axis equal;
box;

xlabel('Comp. 1');
ylabel('Comp. 2');


%% Create transition matrix and order it
transition_index_vec = clustering_labels(1:end - 1) + (clustering_labels(2:end) - 1) * NUMBER_OF_CLUSTERS;
[transition_index_count, ~] = histcounts(transition_index_vec, [0.5:1:NUMBER_OF_CLUSTERS^2 + 0.5]);
transition_index_mat = reshape(transition_index_count, [NUMBER_OF_CLUSTERS NUMBER_OF_CLUSTERS])';

transition_mat = transition_index_mat ./ repmat(sum(transition_index_mat, 2), [1 NUMBER_OF_CLUSTERS]);

% Run ordering code to get correct shuffling of the matrix
Ordering_cyclical_Ver000

ordered_clustering_labels = zeros(size(clustering_labels));

for cluster_index = 1:NUMBER_OF_CLUSTERS
    cluster_indices = find(clustering_labels == chosen_shuffle(cluster_index));
    
    if MIRROR_ORDERING == true
        ordered_clustering_results(cluster_indices) = NUMBER_OF_CLUSTERS + 1 - cluster_index;
    else
        ordered_clustering_results(cluster_indices) = cluster_index;
    end
end

estimated_angle_by_clustering = mod(CENTER_OF_CLUSTERING_ANGLE_BINS(ordered_clustering_results) + ACTUAL_VERSUS_CLUSTERING_SHIFT, 2 * pi);

%% Truncate data according to actual frames being used
full_estimated_angle_by_clustering = zeros(size(full_neuron_firing_per_bin, 1), 1);
full_estimated_angle_by_clustering(filter_mask) = estimated_angle_by_clustering;

smoothed_estimated_angle_by_clustering = smooth_estimated_angle(full_estimated_angle_by_clustering, filter_mask)';

%% Create tuning curves by clusters
neuron_by_cluster_spike_count = zeros(NUMBER_OF_CLUSTERS, number_of_neurons);

% Count the number of frames of each cluster
frames_per_cluster_count = histcounts(clustering_labels,  0.5:1:NUMBER_OF_CLUSTERS + 0.5);

for cluster_index = 1:NUMBER_OF_CLUSTERS
    cluster_frames_indices = find(clustering_labels == cluster_index);
    
    neuron_by_cluster_spike_count(cluster_index, :) = sum(filtered_neuron_firing(cluster_frames_indices, :), 1);
end

neuron_firing_rate = (neuron_by_cluster_spike_count ./ repmat(frames_per_cluster_count', [1 number_of_neurons])) * (BEHAVIORAL_SAMPLE_RATE / BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN);

ordered_neuron_firing_rate = neuron_firing_rate(chosen_shuffle, :);


%% Clustering results plotting

% Plot clustering histograms versus actual head direction
figure;
for figure_index = 1:NUMBER_OF_CLUSTERS
    subplot(NUMBER_OF_CLUSTERS, 1, figure_index);
    hist(filtered_angle_per_temporal_bin(clustering_labels == figure_index), CENTER_OF_ANGLE_BINS);
end


%% Polar plot tuning curves created by clustering (all neurons)
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

%% Plot the estimated polar plots
mouse = 'Mouse28';
mouse = mouse_by_electrode_brain_region(mouse);
id = mouse('140313');

firing_rate = create_firing_rate_matrix(filtered_full_neuron_firing_per_bin, smoothed_estimated_angle_by_clustering);
plot_polar_tuning_curve(firing_rate, head_direction_neurons_indices, Map, id);
plot_polar_tuning_curve(spike_rate_mat_neuron_by_angle, head_direction_neurons_indices, Map, id);

% Calculate correlation over actual and inferred tuning curve
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

colormap jet;
bar(CENTER_OF_HISTOGRAM_BINS, hist_mat', 'stacked');

xlabel('Correlation');
ylabel('Count');

%%
persistent_topology;