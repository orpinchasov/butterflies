%% Configuration and globals

constants;

TYPE = 'wake';
FILTERED = true;


% Example neurons for panel G for Mouse28-140313 thalamus wake
%EXAMPLE_NEURONS = [3 4 5 6 17 19 20 22];
%EXAMPLE_NEURONS = [3 4 6 9 14 15 19 20];

NUMBER_OF_REDUCED_DIMENSIONS_FOR_PCA = 5;

if strcmp(TYPE, 'wake')
    load('reduced_data_rem_all');
    load('index_of_visualization_angle_per_temporal_bin_rem_all');
end

if strcmp(TYPE, 'wake') && FILTERED == false
    SLOPE_MULTIPLIER = -1;
    CORRECTED_NEURON_PREFERRED_ANGLE_SHIFT = 1.5 * pi;
    
    % Example neurons for panel G for Mouse28-140313 all brain wake
    % two directions:
    % 45 47
    % 
    % one direction:
    % 43 46 55 56 61
    % 
    % no direction:
    % 62
    EXAMPLE_NEURONS = [43 45 47 62 55 56 46 61];
    
    PANEL_A_XLIM = [-0.012 0.015];
    PANEL_A_YLIM = [-0.013 0.014];
    
    PANEL_D_XLIM = [-0.7 0.7];
    PANEL_D_YLIM = [-0.8 0.6];
elseif strcmp(TYPE, 'wake') && FILTERED == true
    SLOPE_MULTIPLIER = -1;
    CORRECTED_NEURON_PREFERRED_ANGLE_SHIFT = 0.85 * pi;
    
    PANEL_A_XLIM = [-0.015 0.012];
    PANEL_A_YLIM = [-0.012 0.015];

    PANEL_D_XLIM = [-0.8 0.8];
    PANEL_D_YLIM = [-0.6 1.0];
    
    EXAMPLE_NEURONS = [43 45 47 62 55 56 46 61];
elseif strcmp(TYPE, 'rem') && FILTERED == false
    SLOPE_MULTIPLIER = 1;
    CORRECTED_NEURON_PREFERRED_ANGLE_SHIFT = -1.1 * pi;
    
    % Mouse28-140313 all rem
    EXAMPLE_NEURONS = [46 45 57 62 59 44 55 24];

    PANEL_A_XLIM = [-0.018 0.017];
    PANEL_A_YLIM = [-0.016 0.019];
    
    PANEL_D_XLIM = [-0.7 0.7];
    PANEL_D_YLIM = [-0.8 0.6];
elseif strcmp(TYPE, 'rem') && FILTERED == true
    SLOPE_MULTIPLIER = 1;
    CORRECTED_NEURON_PREFERRED_ANGLE_SHIFT = -0.3 * pi;
    
    % Mouse28-140313 all rem
    EXAMPLE_NEURONS = [46 45 57 62 59 44 55 24];

    PANEL_A_XLIM = [-0.018 0.017];
    PANEL_A_YLIM = [-0.016 0.019];
    
    PANEL_D_XLIM = [-0.7 0.7];
    PANEL_D_YLIM = [-0.8 0.6];
else
    warning('Unknown combination of parameters!');
end

figure;
findfigs;

%Some WYSIWYG options:
set(gcf, 'DefaultAxesFontSize', 8);
set(gcf, 'DefaultAxesFontName', 'arial');
fig_size_y = 25;
fig_size_x = 22;

set(gcf, 'PaperUnits', 'centimeters', 'PaperPosition', [5 2 fig_size_x fig_size_y]);
set(gcf, 'PaperOrientation', 'portrait');
set(gcf, 'Units', 'centimeters', 'Position', get(gcf,'paperPosition') + [3 -6 0 0]);

% Define sizes and spacings
x_size1 = 0.14;
x_size2 = 0.01; % colorbar
x_size_insert = 0.1;

y_size1 = x_size1 * fig_size_x / fig_size_y;
y_size2 = y_size1;
y_size_insert = x_size_insert * fig_size_x / fig_size_y;

x_margin = 0.06;
x_spacing = 0.24;
y_margin = 0.04;
y_spacing = 0.2;

grid_1_x = x_margin;
grid_2_x = grid_1_x + x_spacing;
grid_3_x = grid_2_x + x_spacing;
grid_4_x = grid_3_x + x_spacing;

grid_1_y = y_margin;
grid_2_y = grid_1_y + y_spacing;
grid_3_y = grid_2_y + y_spacing;
grid_4_y = grid_3_y + y_spacing;
grid_5_y = grid_4_y + y_spacing;


% Panel A - reduced neuronal data projected onto a 2D plane
axes('position', [grid_1_x grid_5_y x_size1 y_size1]);

% Plot the angle on the reduced data
head_direction_cmap = hsv(NUMBER_OF_ANGLE_BINS);
% Add black color for 'nan' values
head_direction_cmap = [head_direction_cmap; 0 0 0];

index_of_visualization_angle_per_temporal_bin = round(NUMBER_OF_ANGLE_BINS * visualization_angle_per_temporal_bin / ( 2 * pi));
index_of_visualization_angle_per_temporal_bin(index_of_visualization_angle_per_temporal_bin == 0) = NUMBER_OF_ANGLE_BINS;
% Color the missing values in black
index_of_visualization_angle_per_temporal_bin(isnan(index_of_visualization_angle_per_temporal_bin)) = NUMBER_OF_ANGLE_BINS + 1;

scatter(reduced_data(:, 2), reduced_data(:, 3), 1, 'fill', 'k');

set(gca, 'ytick', [-0.01 0 0.01]);

xlabel('Comp. 1');
ylabel('Comp. 2');

xlim(PANEL_A_XLIM);
ylim(PANEL_A_YLIM);

% Panel B - plot averaged clustered data 
axes('position', [grid_2_x grid_5_y x_size1 y_size1]);
hold on;

% (in order to compare with the transition graph)

% The following mechanism allows rotating the color map to match that of
% the reduced data.
MIRROR = -1;
OFFSET = 3;

angle_bins_cmap = hsv(NUMBER_OF_ANGLE_BINS);
clusters_indices = NUMBER_OF_ANGLE_BINS / NUMBER_OF_CLUSTERS:NUMBER_OF_ANGLE_BINS / NUMBER_OF_CLUSTERS:NUMBER_OF_ANGLE_BINS;
clusters_cmap = angle_bins_cmap(1 + mod(MIRROR * clusters_indices + OFFSET * NUMBER_OF_ANGLE_BINS / NUMBER_OF_CLUSTERS, 40), :);

average_cluster_point = ones(NUMBER_OF_CLUSTERS, 2);
for cluster_index = 1:NUMBER_OF_CLUSTERS
    cluster_indices = find(clustering_labels == chosen_shuffle(cluster_index));
    average_cluster_point(cluster_index, :) = mean(reduced_data(cluster_indices, 2:3));
end

scatter(reduced_data(:, 2), reduced_data(:, 3), 1, 'fill', 'k', 'MarkerFaceAlpha', 0.05, 'MarkerEdgeAlpha', 0.05);
scatter(average_cluster_point(:, 1), average_cluster_point(:, 2), 100, clusters_cmap, 'fill', 'MarkerFaceAlpha', 0.7, 'MarkerEdgeAlpha', 0.7);

set(gca, 'ytick', [-0.01 0 0.01]);

xlim(PANEL_A_XLIM);
ylim(PANEL_A_YLIM);

% Mouse28-140313 all wake unfiltered
%xlim([-0.012 0.015]);
%ylim([-0.013 0.014]);

% Mouse28-140313 all wake filtered
%xlim([-0.015 0.012]);
%ylim([-0.012 0.015]);

% Mouse 28-140313 thalamus wake
%xlim([-0.013 0.013]);
%ylim([-0.016 0.010]);

% Mouse 28-140313 all rem
%xlim([-0.016 0.015]);
%ylim([-0.014 0.017]);

xlabel('Comp. 1');
ylabel('Comp. 2');

% Panel C - ordered transition matrix
axes('position', [grid_3_x * 0.94 grid_5_y x_size1 y_size1]);

imagesc(imcomplement(transition_mat(chosen_shuffle, chosen_shuffle)));
set(gca, 'xticklabel', []);
set(gca, 'yticklabel', []);

xlabel('Cluster (t+1)');
ylabel('Cluster (t)');

% Panel C - colorbar
axes('position', [grid_3_x * 0.95 + x_size1 + 0.01 grid_5_y x_size2 y_size2], 'YAxisLocation', 'right');
hold on;

cmap_jet=1-colormap('gray');
num_colors=size(cmap_jet,1);
set(gca,'xtick',[])
xlim([0 1])
ylim([0 1])
for n=1:num_colors
    p=patch([0 1 1 0],[n/num_colors n/num_colors (n-1)/num_colors (n-1)/num_colors],cmap_jet(n,:));
    set(p,'FaceAlpha',1,'EdgeColor','none');
end

plot([0 1 1 0 0], [0 0 1 1 0], 'k-');
set(gca,'ytick',[0 1])
set(gca, 'yticklabel', [0 1]);

y = ylabel('Transition probability');
set(y, 'Units', 'Normalized', 'Position', [2.5, 0.5, 0]);

% Panel D - plot transition probability graph
axes('position', [grid_4_x grid_5_y x_size1 y_size1]);

% The following mechanism allows rotating the color map to match that of
% the reduced data.
MIRROR = 1;
OFFSET = 5;

rng(0);

% PCA over transition probability
ordered_transition_mat = transition_mat(chosen_shuffle, chosen_shuffle);

%W = max(ordered_transition_mat, ordered_transition_mat');
W = (ordered_transition_mat + ordered_transition_mat') / 2;

% The original normalized graph Laplacian, non-corrected for density
ld = diag(sum(W,2).^(-1/2));
DO = ld*W*ld;
DO = max(DO,DO');%(DO + DO')/2;

% get eigenvectors
[v,d] = eigs(DO, NUMBER_OF_REDUCED_DIMENSIONS_FOR_PCA, 'la');

% 'v' also known as 'transition_matrix_states'
angle_bins_cmap = hsv(NUMBER_OF_ANGLE_BINS);
clusters_indices = NUMBER_OF_ANGLE_BINS / NUMBER_OF_CLUSTERS:NUMBER_OF_ANGLE_BINS / NUMBER_OF_CLUSTERS:NUMBER_OF_ANGLE_BINS;
clusters_cmap = angle_bins_cmap(1 + mod(MIRROR * clusters_indices + OFFSET * NUMBER_OF_ANGLE_BINS / NUMBER_OF_CLUSTERS, 40), :);
scatter(-v(:, 2), v(:, 3), 100, clusters_cmap, 'fill');

xlim(PANEL_D_XLIM);
ylim(PANEL_D_YLIM);

% Mouse28-140313 all wake
%xlim([-0.7 0.7]);
%ylim([-0.6 0.8]);

% Mouse28-140313 all wake unfiltered
%xlim([-0.7 0.7]);
%ylim([-0.8 0.6]);

% Mouse28-140313 thalamus wake
%xlim([-0.6 0.8]);
%ylim([-0.8 0.6]);

xlabel('Comp. 1');
ylabel('Comp. 2');


% Panel E - trajectory of actual head movement versus clustered movement
axes('position', [grid_1_x grid_4_y * 0.98 x_size1 * 2 + x_spacing - x_size1 y_size1]);

hold on;

if strcmp(TYPE, 'wake')
    plot(filtered_angle_per_temporal_bin, 'k.');
elseif strcmp(TYPE, 'rem')
    plot(estimated_head_direction_angle_per_sample_index, 'k.');
else
    warning('Unknown type!')
end

plot(smoothed_estimated_angle_by_clustering, 'r.');

ylim([0 2 * pi]);
xlim([2000 6000]);

% This is stupid
ticks = get(gca,'XTick');
set(gca, 'XTickLabel', cellstr(num2str(round(ticks / 10)')));

xlabel('Time (samples)');
ylabel('Angle (rad)');

if strcmp(TYPE, 'wake')
    [l,l2,l3,l4]=legend('Head direction', 'Internal direction', 'Location', 'northwest', 'orientation','horizontal');

    for n=1:length(l2)
        if sum(strcmp(properties(l2(n)),'MarkerSize'))
            l2(n).MarkerSize=20;
        end
    end
    
    set(l, 'position', get(l, 'position') + [-0.03 0.033 0 0]);
    set(l, 'fontsize', 8);
elseif strcmp(TYPE, 'rem')
    l = legend('Estimated HD', 'Internal direction', 'Location', 'northwest', 'orientation','horizontal');
    set(l, 'position', get(l, 'position') + [-0.03 0.033 0 0]);
    set(l, 'fontsize', 8);
else
    warning('Unknown type!')
end

legend boxoff;

% Panel E (inset) - estimated head direction as decoder's output vs
% smoothed estimated angle by clustering
axes('position', [grid_2_x + x_size1 * 0.37 grid_4_y + y_size1 * 0.25 x_size_insert y_size_insert]);
scatter(estimated_head_direction_angle_per_sample_index, smoothed_estimated_angle_by_clustering, 1, 'k.');
diffs = estimated_head_direction_angle_per_sample_index - smoothed_estimated_angle_by_clustering;
histogram(mod(diffs + 1 * pi, 2*pi) - 1 * pi, 30, 'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 1, 'normalization', 'pdf');
xlim([-pi pi]);
ylim([0 1]);

set(gca, 'xtick', [-pi pi]);
set(gca, 'XTickLabel', {'-\pi', '\pi'});
set(gca, 'ytick', [0 1]);
set(gca, 'yTickLabel', {'0', '1'});
text(7.6, 7.6, '2\pi', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'fontsize', 8);
set(gca,'xaxisLocation', 'top');
set(gca,'yaxisLocation', 'right');
l = xlabel('Decoder error (rad)');
set(l, 'position', get(l, 'position') + [0 -0.05 0]);
l = ylabel('Fraction');
set(l, 'position', get(l, 'position') + [-0.03 0 0]);

box;

% Panel F - trajectory of actual head movement versus clustered movement
plot_polar_tuning_curve_by_axes(spike_rate_mat_neuron_by_angle, firing_rate, EXAMPLE_NEURONS, grid_3_x, grid_3_y * 0.93, 0.10, 0.09);

% Panel G - actual angle per temporal bin over reduced data
axes('position', [grid_1_x grid_3_y x_size1 y_size1]);

% Plot the angle on the reduced data
cmap3 = hsv(NUMBER_OF_ANGLE_BINS);
% Add black color for 'nan' values
cmap3 = [cmap3; 0 0 0];

if strcmp(TYPE, 'wake')
    visualization_angle_per_temporal_bin = filtered_angle_per_temporal_bin;
elseif strcmp(TYPE, 'rem')
    visualization_angle_per_temporal_bin = estimated_head_direction_angle_per_sample_index;
else
    warning('Unknown type!')
end


index_of_visualization_angle_per_temporal_bin = round(NUMBER_OF_ANGLE_BINS * visualization_angle_per_temporal_bin / ( 2 * pi));
index_of_visualization_angle_per_temporal_bin(index_of_visualization_angle_per_temporal_bin == 0) = NUMBER_OF_ANGLE_BINS;
% Color the missing values in black
index_of_visualization_angle_per_temporal_bin(isnan(index_of_visualization_angle_per_temporal_bin)) = NUMBER_OF_ANGLE_BINS + 1;

scatter(reduced_data(:, 2), reduced_data(:, 3), 1, cmap3(index_of_visualization_angle_per_temporal_bin, :), 'fill');

set(gca, 'ytick', [-0.01 0 0.01]);

title('Head direction');

xlabel('Comp. 1');
ylabel('Comp. 2');

xlim(PANEL_A_XLIM);
ylim(PANEL_A_YLIM);

% Mouse28-140313 all wake unfiltered
%xlim([-0.012 0.015]);
%ylim([-0.013 0.014]);

% Mouse28-140313 all wake filtered
%xlim([-0.015 0.012]);
%ylim([-0.012 0.015]);

% Panel H - Estimated clustered angle over reduced data

axes('position', [grid_2_x grid_3_y x_size1 y_size1]);

visualization_angle_per_temporal_bin = smoothed_estimated_angle_by_clustering;

index_of_visualization_angle_per_temporal_bin = round(NUMBER_OF_ANGLE_BINS * visualization_angle_per_temporal_bin / ( 2 * pi));
index_of_visualization_angle_per_temporal_bin(index_of_visualization_angle_per_temporal_bin == 0) = NUMBER_OF_ANGLE_BINS;
% Color the missing values in black
index_of_visualization_angle_per_temporal_bin(isnan(index_of_visualization_angle_per_temporal_bin)) = NUMBER_OF_ANGLE_BINS + 1;

scatter(reduced_data(:, 2), reduced_data(:, 3), 1, cmap3(index_of_visualization_angle_per_temporal_bin, :), 'fill');

set(gca, 'ytick', [-0.01 0 0.01]);

title('Internal direction');

xlabel('Comp. 1');
ylabel('Comp. 2');

xlim(PANEL_A_XLIM);
ylim(PANEL_A_YLIM);

% Mouse28-140313 all wake unfiltered
%xlim([-0.012 0.015]);
%ylim([-0.013 0.014]);

% Mouse28-140313 all wake filtered
%xlim([-0.015 0.012]);
%ylim([-0.012 0.015]);

% Panel G - scattering of clustering tuning curve versus actual tuning curve
axes('position', [grid_1_x grid_2_y * 1.03 x_size1 y_size1]);

neuron_actual_preferred_angle = zeros(number_of_neurons, 1);
neuron_clustering_preferred_angle = zeros(number_of_neurons, 1);

for neuron_index = 1:number_of_neurons
    current_neuron_spike_rate_by_angle = spike_rate_mat_neuron_by_angle(neuron_index, :);

    neuron_actual_preferred_angle(neuron_index) = angle(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_ANGLE_BINS)));
end

for neuron_index = 1:number_of_neurons
    current_neuron_spike_rate_by_angle = ordered_neuron_firing_rate(:, neuron_index)';

    neuron_clustering_preferred_angle(neuron_index) = angle(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_CLUSTERING_ANGLE_BINS)));
end

corrected_neuron_clustering_preferred_angle = mod(SLOPE_MULTIPLIER * (neuron_clustering_preferred_angle + CORRECTED_NEURON_PREFERRED_ANGLE_SHIFT), 2 * pi);

head_direction_neurons_indices = find_head_direction_neurons(spike_rate_mat_neuron_by_angle);

hold on;

plot([0 2 * pi], [0 2 * pi], 'r-');
h1 = scatter(corrected_neuron_clustering_preferred_angle, mod(neuron_actual_preferred_angle, 2 * pi), 10, 'k');
h2 = scatter(corrected_neuron_clustering_preferred_angle(head_direction_neurons_indices), mod(neuron_actual_preferred_angle(head_direction_neurons_indices), 2 * pi), 10, 'fill', 'k');

axis equal;

xlim([0 2 * pi]);
ylim([0 2 * pi]);

xlabel('Internal preferred direction (rad)', 'FontSize', 8);
ylabel('Preferred HD (rad)', 'FontSize', 8);

l = legend([h1 h2], 'Non-HD cells', 'HD cells', 'location', 'northwest', 'orientation','horizontal');
set(l, 'position', get(l, 'position') + [-0.06 0.035 0 0]);
legend boxoff;

% Panel H2 - same as above for Reighly vector length
axes('position',  [grid_2_x grid_2_y * 1.03 x_size1 y_size1]);

neuron_actual_vector_length = zeros(number_of_neurons, 1);
neuron_clustering_vector_length = zeros(number_of_neurons, 1);

for neuron_index = 1:number_of_neurons
    current_neuron_spike_rate_by_angle = spike_rate_mat_neuron_by_angle(neuron_index, :);

    neuron_actual_vector_length(neuron_index) = abs(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_ANGLE_BINS))) ./ sum(abs(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_ANGLE_BINS)));
end

for neuron_index = 1:number_of_neurons
    current_neuron_spike_rate_by_angle = ordered_neuron_firing_rate(:, neuron_index)';

    neuron_clustering_vector_length(neuron_index) = abs(sum(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_CLUSTERING_ANGLE_BINS))) ./ sum(abs(current_neuron_spike_rate_by_angle .* exp(1i * CENTER_OF_CLUSTERING_ANGLE_BINS)));
end

hold on;

plot([0 2 * pi], [0 2 * pi], 'r-');
h1 = scatter(neuron_clustering_vector_length, neuron_actual_vector_length, 10, 'k');
h2 = scatter(neuron_clustering_vector_length(head_direction_neurons_indices), neuron_actual_vector_length(head_direction_neurons_indices), 10, 'fill', 'k');
axis equal;

xlim([0 1]);
ylim([0 1]);

xlabel('Internal vector length', 'FontSize', 8);
ylabel('HD vector length', 'FontSize', 8);

l = legend([h1 h2], 'Non-HD cells', 'HD cells', 'location', 'northwest', 'orientation','horizontal');
set(l, 'position', get(l, 'position') + [-0.06 0.035 0 0]);
legend boxoff;

% Panel I - results accuracy plot
axes('position', [grid_3_x grid_2_y * 1.03 x_size1 y_size1]);

b = bar(CENTER_OF_HISTOGRAM_BINS, hist_mat', 'stacked');

l = legend([b(2) b(1)], 'Non-HD cells', 'HD cells', 'location', 'northwest', 'orientation','horizontal');
set(l, 'position', get(l, 'position') + [-0.06 0.035 0 0]);
legend boxoff;

xlabel('Correlation');
ylabel('Count');

box off;

% Panel J - dimension estimation
axes('position', [grid_4_x grid_2_y * 1.03 x_size1 y_size1]);
start_value=0.004;
end_value=0.023;
slope_one_color=[1 0.2 0.8];

DataForDimEst=full_reduced_data{3}(1:10:end,2:10);
NumOfDataPoints=size(DataForDimEst,1);
RepAndPermData=permute(repmat(DataForDimEst,[1 1 NumOfDataPoints]),[1 3 2]);
diff_RepAndPermData=RepAndPermData-permute(RepAndPermData,[2 1 3]);
diff_RepAndPermData=(sum(diff_RepAndPermData.^2,3)).^0.5;
diff_RepAndPermData(~~eye(NumOfDataPoints))=NaN;

[a b]=hist(diff_RepAndPermData(:),1000);
[~,start_bin_ind]=min(abs(b-start_value));
[~,end_bin_ind]=min(abs(b-end_value));

vy=cumsum(a(start_bin_ind:end));
vx=b(start_bin_ind:end)-b(start_bin_ind-1);
relevant_ind=1:(end_bin_ind-start_bin_ind+1);
hold on;
for run1=6:30
    h1 = plot([-11 -1],run1+1*[-11 -1], '-', 'color', [0.5 0.5 0.5]);
end
% Use the handles for the legend
h2 = plot(log(vx(relevant_ind)), log(vy(relevant_ind)), 'r.');
h3 = plot(log(vx(relevant_ind(end)+1:end)), log(vy(relevant_ind(end)+1:end)), 'r-');

legend([h1 h3], 'slope 1', 'data', 'location', 'northwest');

xlim([-11 -2])
ylim([4 20])

xlabel('Log radius');
ylabel('Log number of neighbors');


if strcmp(TYPE, 'wake')
    axes('position', [grid_1_x grid_1_y x_size1 y_size1]);

    scatter(reduced_data_rem_all(:, 2), reduced_data_rem_all(:, 3), 1, '.k');

    set(gca, 'ytick', [-0.01 0 0.01]);

    xlabel('Comp. 1');
    ylabel('Comp. 2');
    
    title('REM sleep');
    
    xlim([-0.018 0.017]);
    ylim([-0.016 0.019]);
    
    firing_rate_rem = load('firing_rate_rem.mat');
    firing_rate_rem = firing_rate_rem.firing_rate;

    
    axes('position', [grid_2_x grid_1_y * 0.98 x_size1 * 2 + x_spacing - x_size1 y_size1]);
    hold on;
    % Plot the actual head direction from wake data
    plot(filtered_angle_per_temporal_bin, 'k.');

    % Plot the estimated head direction using the rem tuning curves and the
    % filtered full neuron firing per per, which matches the filtered angle.
    smooth_maximum_likelihood_angle_per_sample_index = estimate_head_direction(firing_rate_rem, filtered_full_neuron_firing_per_bin);

    plot(smooth_maximum_likelihood_angle_per_sample_index, 'r.');

    ylim([0 2 * pi]);
    xlim([2000 6000]);
    
    % This is stupid
    ticks = get(gca,'XTick');
    set(gca, 'XTickLabel', cellstr(num2str(round(ticks / 10)')));

    xlabel('Time (samples)');
    ylabel('Angle (rad)');

    if strcmp(TYPE, 'wake')
        [l,l2,l3,l4]=legend('Head direction', 'Decoded direction', 'Location', 'northwest', 'orientation','horizontal');

        for n=1:length(l2)
            if sum(strcmp(properties(l2(n)),'MarkerSize'))
                l2(n).MarkerSize=20;
            end
        end

        set(l, 'position', get(l, 'position') + [-0.03 0.033 0 0]);
        set(l, 'fontsize', 8);
    end

    legend boxoff;
    
    axes('position', [grid_3_x + x_size1 * 0.37 grid_1_y + y_size1 * 0.35 x_size_insert y_size_insert]);
    diffs = smooth_maximum_likelihood_angle_per_sample_index - filtered_angle_per_temporal_bin';
    histogram(mod(diffs + 1 * pi, 2*pi) - 1 * pi, 30, 'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 1, 'normalization', 'pdf');
    xlim([-pi pi]);
    ylim([0 1]);

    set(gca, 'xtick', [-pi pi]);
    set(gca, 'XTickLabel', {'-\pi', '\pi'});
    set(gca, 'ytick', [0 1]);
    set(gca, 'yTickLabel', {'0', '1'});
    text(7.6, 7.6, '2\pi', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'fontsize', 8);
    set(gca,'xaxisLocation', 'top');
    set(gca,'yaxisLocation', 'right');
    l = xlabel('Decoder error (rad)');
    set(l, 'position', get(l, 'position') + [0 -0.05 0]);
    l = ylabel('Fraction');
    set(l, 'position', get(l, 'position') + [-0.03 0 0]);

    box;
else
    % Panel J - persistent topology
    axes('position', [grid_1_x grid_1_y x_size1 y_size1]);
    hold on;
    number_of_results = size(dimension_0, 1);
    ylim([0 number_of_results + 1]);
    for i = 1:number_of_results
        if dimension_0(i, 2) == Inf
            plot([dimension_0(i, 1) MAX_FILTRATION_VALUE], [number_of_results + 1 - i number_of_results + 1 - i], 'b');
        else
            plot([dimension_0(i, 1) dimension_0(i, 2)], [number_of_results + 1 - i number_of_results + 1 - i], 'b');
        end
    end
    xlim([0 MAX_FILTRATION_VALUE]);

    set(gca, 'xticklabel', []);
    set(gca, 'yticklabel', []);

    xlabel('Radius');
    ylabel('Index');

    title('\beta_0 (components)');

    axes('position', [grid_2_x grid_1_y x_size1 y_size1]);
    hold on;
    number_of_results = size(dimension_1, 1);
    ylim([0 number_of_results + 1]);
    for i = 1:number_of_results
        if dimension_1(i, 2) == Inf
            plot([dimension_1(i, 1) MAX_FILTRATION_VALUE], [number_of_results + 1 - i number_of_results + 1 - i], 'b');
        else
            plot([dimension_1(i, 1) dimension_1(i, 2)], [number_of_results + 1 - i number_of_results + 1 - i], 'b');
        end
    end
    xlim([0 MAX_FILTRATION_VALUE]);

    set(gca, 'xticklabel', []);
    set(gca, 'yticklabel', []);

    xlabel('Radius');
    ylabel('Index');

    title('\beta_1 (holes)');

    axes('position', [grid_3_x grid_1_y x_size1 y_size1]);
    hold on;
    xlim([0 MAX_FILTRATION_VALUE]);

    set(gca, 'xticklabel', []);
    set(gca, 'yticklabel', []);

    xlabel('Radius');
    ylabel('Index');

    title('\beta_2 (spaces)');
end
    
colormap gray;

set(gcf,'PaperPositionMode','auto');