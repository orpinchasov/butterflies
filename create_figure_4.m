%% Configuration and globals

constants;

% Example neurons for panel G for Mouse28-140313
% two directions:
% 45 47
% 
% one direction:
% 43 46 55 56 61
% 
% no direction:
% 62
EXAMPLE_NEURONS = [43 46 55 56 61 45 47 62];

NUMBER_OF_REDUCED_DIMENSIONS_FOR_PCA = 5;

%% Panel A - reduced neuronal data projected onto a 2D plane
% Plot the angle on the reduced data
head_direction_cmap = hsv(NUMBER_OF_ANGLE_BINS);
% Add black color for 'nan' values
head_direction_cmap = [head_direction_cmap; 0 0 0];

index_of_visualization_angle_per_temporal_bin = round(NUMBER_OF_ANGLE_BINS * visualization_angle_per_temporal_bin / ( 2 * pi));
index_of_visualization_angle_per_temporal_bin(index_of_visualization_angle_per_temporal_bin == 0) = NUMBER_OF_ANGLE_BINS;
% Color the missing values in black
index_of_visualization_angle_per_temporal_bin(isnan(index_of_visualization_angle_per_temporal_bin)) = NUMBER_OF_ANGLE_BINS + 1;

figure(1);
scatter(reduced_data(:, 2), reduced_data(:, 3), 5, head_direction_cmap(index_of_visualization_angle_per_temporal_bin, :), 'fill');

xlabel('Comp. 1');
ylabel('Comp. 2');

%% Panel C - ordered transition matrix
figure; 
colormap('jet');
imagesc(transition_mat(chosen_shuffle, chosen_shuffle));
axis square;
caxis([0 1]);
colorbar;

%% Panel D - plot averaged clustered data 
% (in order to compare with the transition graph)
clusters_cmap = hsv(NUMBER_OF_CLUSTERS);

average_cluster_point = ones(NUMBER_OF_CLUSTERS, 2);
for cluster_index = 1:NUMBER_OF_CLUSTERS
    cluster_indices = find(clustering_labels == chosen_shuffle(cluster_index));
    average_cluster_point(cluster_index, :) = mean(reduced_data(cluster_indices, 2:3));
end

figure; scatter(average_cluster_point(:, 1), average_cluster_point(:, 2), 300, clusters_cmap, 'fill');

axis equal;
box;

xlim([-0.012 0.015]);
ylim([-0.013 0.014]);

%% Panel E - plot transition probability graph
% PCA over transition probability
ordered_transition_mat = transition_mat(chosen_shuffle, chosen_shuffle);

W = max(ordered_transition_mat, ordered_transition_mat');

% The original normalized graph Laplacian, non-corrected for density
ld = diag(sum(W,2).^(-1/2));
DO = ld*W*ld;
DO = max(DO,DO');%(DO + DO')/2;

% get eigenvectors
[v,d] = eigs(DO, NUMBER_OF_REDUCED_DIMENSIONS_FOR_PCA, 'la');

% 'v' also known as 'transition_matrix_states'
figure;

scatter(-v(:, 2), v(:, 3), 300, clusters_cmap, 'fill');

axis equal;
box;

xlim([-0.7 0.7]);
ylim([-0.8 0.6]);

%% Panel G - actual versus estimated head direction polar plot

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

%% Panel H - scattering of clustering tuning curve versus actual tuning curve
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

plot([0 2 * pi], [0 2 * pi], 'r-');
scatter(mod(neuron_actual_preferred_angle, 2 * pi), corrected_neuron_clustering_preferred_angle, 'b');
scatter(mod(neuron_actual_preferred_angle(head_direction_neurons_indices), 2 * pi), corrected_neuron_clustering_preferred_angle(head_direction_neurons_indices), 'fill', 'b');

axis equal;

xlim([0 2 * pi]);
ylim([0 2 * pi]);

xlabel('Preferred head direction (rad)');
ylabel('Reconstructed preferred head direction (rad)');

%% Panel H2 - same as above for Reighly vector length

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

%% Panel I - dimension estimation
RingDimEst_Ver000

%% Panel J - persistent topology
% TODO: Currently requires separate handling
%persistent_topology