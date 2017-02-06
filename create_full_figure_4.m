%% Configuration and globals

constants;

% Example neurons for panel G for Mouse28-140313 all brain wake
% two directions:
% 45 47
% 
% one direction:
% 43 46 55 56 61
% 
% no direction:
% 62
EXAMPLE_NEURONS = [43 45 47 62 46 56 55 61];

% Mouse28-140313 all rem
%EXAMPLE_NEURONS = [44 46 55 59 60 45 57 62];

% Example neurons for panel G for Mouse28-140313 thalamus wake
%EXAMPLE_NEURONS = [3 4 5 6 17 19 20 22];
%EXAMPLE_NEURONS = [3 4 6 9 14 15 19 20];

NUMBER_OF_REDUCED_DIMENSIONS_FOR_PCA = 5;

SLOPE_MULTIPLIER = -1; % Either 1 or -1 to turn slope
% Filtered data
CORRECTED_NEURON_PREFERRED_ANGLE_SHIFT = 0.85 * pi;
% Unfiltered data
%CORRECTED_NEURON_PREFERRED_ANGLE_SHIFT = 1.5 * pi;

figure;
findfigs;

%Some WYSIWYG options:
set(gcf,'DefaultAxesFontSize',8);
set(gcf,'DefaultAxesFontName','arial');
fig_size_y=25;
fig_size_x=22;
set(gcf,'PaperUnits','centimeters','PaperPosition',[5 2 fig_size_x fig_size_y]);
set(gcf,'PaperOrientation','portrait');
set(gcf,'Units','centimeters','Position',get(gcf,'paperPosition')+[3 -6 0 0]); %+[4 2 0 0]);

% Define sizes and spacings
x_size1=0.18; %
x_size2=0.11; %
x_size3=0.25; %
x_size4=0.4; %
x_size5=0.02; % colorbar
x_size_insert = 0.1;
x_space1=0.07;
x_space2=0.025;

y_size1=x_size1*fig_size_x/fig_size_y;
y_size2=x_size2*fig_size_x/fig_size_y;
y_size3=0.125;
y_size4=0.32;
y_size5=y_size1;
y_size_insert = x_size_insert*fig_size_x/fig_size_y;
y_space1=0.1;
y_space2=0.1;

x_margin = 0.06;
x_spacing = 0.24;
y_margin = 0.02;
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

scatter(reduced_data(:, 2), reduced_data(:, 3), 5, 'fill');

set(gca, 'xticklabel', []);
set(gca, 'yticklabel', []);

xlabel('Comp. 1');
ylabel('Comp. 2');

% Mouse28-140313 all wake unfiltered
%xlim([-0.012 0.015]);
%ylim([-0.013 0.014]);

% Mouse28-140313 all wake filtered
xlim([-0.015 0.012]);
ylim([-0.012 0.015]);


% Panel B - ordered transition matrix
axes('position', [grid_2_x * 0.92 grid_5_y x_size1 y_size1]);

imagesc(imcomplement(transition_mat(chosen_shuffle, chosen_shuffle)));
set(gca, 'xticklabel', []);
set(gca, 'yticklabel', []);

xlabel('Cluster (t+1)');
ylabel('Cluster (t)');

% Panel B - colorbar
axes('position', [grid_2_x * 0.9 + x_size1 + 0.01 grid_5_y x_size5 y_size5], 'YAxisLocation', 'right');
cmap_jet=1-colormap('gray');
num_colors=size(cmap_jet,1);
set(gca,'xtick',[])
%set(gca,'ytick',[])
xlim([0 1])
ylim([0 1])
for n=1:num_colors
    hold on
    p=patch([0 1 1 0],[n/num_colors n/num_colors (n-1)/num_colors (n-1)/num_colors],cmap_jet(n,:));
    set(p,'FaceAlpha',1,'EdgeColor','none');
end
box;

y = ylabel('Transition probability', 'rot', -90);
set(y, 'Units', 'Normalized', 'Position', [3.5, 0.5, 0]);

% Panel C - plot averaged clustered data 
axes('position', [grid_3_x * 1.05 grid_5_y x_size1 y_size1]);

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

scatter(average_cluster_point(:, 1), average_cluster_point(:, 2), 100, clusters_cmap, 'fill');

%axis equal;
box;

% Mouse28-140313 all wake unfiltered
%xlim([-0.012 0.015]);
%ylim([-0.013 0.014]);

% Mouse28-140313 all wake filtered
xlim([-0.015 0.012]);
ylim([-0.012 0.015]);

% Mouse 28-140313 thalamus wake
%xlim([-0.013 0.013]);
%ylim([-0.016 0.010]);

% Mouse 28-140313 all rem
%xlim([-0.016 0.015]);
%ylim([-0.014 0.017]);

set(gca, 'xticklabel', []);
set(gca, 'yticklabel', []);

xlabel('Comp. 1');
ylabel('Comp. 2');

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

%axis equal;
box;

% Mouse28-140313 all wake
xlim([-0.7 0.7]);
ylim([-0.6 0.8]);

% Mouse28-140313 all wake unfiltered
%xlim([-0.7 0.7]);
%ylim([-0.8 0.6]);

% Mouse28-140313 thalamus wake
%xlim([-0.6 0.8]);
%ylim([-0.8 0.6]);

% Mouse 28-140313 all rem
%xlim([-0.7 0.5]);
%ylim([-0.45 0.75]);

set(gca, 'xticklabel', []);
set(gca, 'yticklabel', []);

xlabel('Comp. 1');
ylabel('Comp. 2');


% Panel E - trajectory of actual head movement versus clustered movement
axes('position', [grid_1_x grid_4_y x_size1 * 2 + x_spacing - x_size1 y_size1]);

%scatter(filtered_angle_per_temporal_bin, smoothed_estimated_angle_by_clustering, 'k.');
%xlim([0 2 * pi]);
%ylim([0 2 * pi]);
%axis square;

hold on;

plot(filtered_angle_per_temporal_bin, 'k.');
plot(smoothed_estimated_angle_by_clustering, 'r.');
%scatter(angle_per_temporal_bin, estimated_angle_by_clustering, '.');

ylim([0 2 * pi]);
xlim([2000 6000]);

% This is stupid
ticks = get(gca,'XTick');
set(gca, 'XTickLabel', cellstr(num2str(round(ticks / 10)')));

xlabel('Sample');
ylabel('Angle (rad)');

legend('Actual head direction', 'Internal head direction', 'Location', 'northwest');

% Panel E (insert) - estimated head direction as decoder's output vs
% smoothed estimated angle by clustering
axes('position', [grid_2_x + x_size1 * 0.5 grid_4_y + y_size1 * 0.5 x_size_insert y_size_insert]);
scatter(estimated_head_direction_angle_per_sample_index, smoothed_estimated_angle_by_clustering, 1, 'k.');
xlim([0 2 * pi]);
ylim([0 2 * pi]);
set(gca, 'XTickLabel', []);
set(gca, 'YTickLabel', []);
box;

% Panel F - trajectory of actual head movement versus clustered movement
%axes('position',[start_x_F start_y_F size_x_F size_y_F]);
plot_polar_tuning_curve_by_axes(spike_rate_mat_neuron_by_angle, firing_rate, EXAMPLE_NEURONS, grid_3_x, grid_3_y, 0.10, 0.09);

% Panel G - actual angle per temporal bin over reduced data
axes('position', [grid_1_x grid_3_y x_size1 y_size1]);

% Plot the angle on the reduced data
cmap3 = hsv(NUMBER_OF_ANGLE_BINS);
% Add black color for 'nan' values
cmap3 = [cmap3; 0 0 0];

visualization_angle_per_temporal_bin = filtered_angle_per_temporal_bin;

index_of_visualization_angle_per_temporal_bin = round(NUMBER_OF_ANGLE_BINS * visualization_angle_per_temporal_bin / ( 2 * pi));
index_of_visualization_angle_per_temporal_bin(index_of_visualization_angle_per_temporal_bin == 0) = NUMBER_OF_ANGLE_BINS;
% Color the missing values in black
index_of_visualization_angle_per_temporal_bin(isnan(index_of_visualization_angle_per_temporal_bin)) = NUMBER_OF_ANGLE_BINS + 1;

scatter(reduced_data(:, 2), reduced_data(:, 3), 5, cmap3(index_of_visualization_angle_per_temporal_bin, :), 'fill');

set(gca, 'xticklabel', []);
set(gca, 'yticklabel', []);

xlabel('Comp. 1');
ylabel('Comp. 2');

% Mouse28-140313 all wake unfiltered
%xlim([-0.012 0.015]);
%ylim([-0.013 0.014]);

% Mouse28-140313 all wake filtered
xlim([-0.015 0.012]);
ylim([-0.012 0.015]);

% Panel H - Estimated clustered angle over reduced data

axes('position', [grid_2_x grid_3_y x_size1 y_size1]);

visualization_angle_per_temporal_bin = smoothed_estimated_angle_by_clustering;

index_of_visualization_angle_per_temporal_bin = round(NUMBER_OF_ANGLE_BINS * visualization_angle_per_temporal_bin / ( 2 * pi));
index_of_visualization_angle_per_temporal_bin(index_of_visualization_angle_per_temporal_bin == 0) = NUMBER_OF_ANGLE_BINS;
% Color the missing values in black
index_of_visualization_angle_per_temporal_bin(isnan(index_of_visualization_angle_per_temporal_bin)) = NUMBER_OF_ANGLE_BINS + 1;

scatter(reduced_data(:, 2), reduced_data(:, 3), 5, cmap3(index_of_visualization_angle_per_temporal_bin, :), 'fill');

set(gca, 'xticklabel', []);
set(gca, 'yticklabel', []);

xlabel('Comp. 1');
ylabel('Comp. 2');

% Mouse28-140313 all wake unfiltered
%xlim([-0.012 0.015]);
%ylim([-0.013 0.014]);

% Mouse28-140313 all wake filtered
xlim([-0.015 0.012]);
ylim([-0.012 0.015]);

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

% Used to get the slope to be positive (bottom left of figure to upper
% right)
%corrected_neuron_clustering_preferred_angle = mod(SLOPE_MULTIPLIER * (neuron_clustering_preferred_angle - ACTUAL_VERSUS_CLUSTERING_SHIFT), 2 * pi);
corrected_neuron_clustering_preferred_angle = mod(SLOPE_MULTIPLIER * (neuron_clustering_preferred_angle + CORRECTED_NEURON_PREFERRED_ANGLE_SHIFT), 2 * pi);

head_direction_neurons_indices = find_head_direction_neurons(spike_rate_mat_neuron_by_angle);

hold on;
box on;

plot([0 2 * pi], [0 2 * pi], 'r-');
scatter(corrected_neuron_clustering_preferred_angle, mod(neuron_actual_preferred_angle, 2 * pi), 10, 'b');
scatter(corrected_neuron_clustering_preferred_angle(head_direction_neurons_indices), mod(neuron_actual_preferred_angle(head_direction_neurons_indices), 2 * pi), 10, 'fill', 'b');

axis equal;

xlim([0 2 * pi]);
ylim([0 2 * pi]);

xlabel('Internal preferred direction (rad)', 'FontSize', 8);
ylabel('Preferred HD (rad)', 'FontSize', 8);

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

%figure;
hold on;
box on;

plot([0 2 * pi], [0 2 * pi], 'r-');
scatter(neuron_clustering_vector_length, neuron_actual_vector_length, 10, 'b');
scatter(neuron_clustering_vector_length(head_direction_neurons_indices), neuron_actual_vector_length(head_direction_neurons_indices), 10, 'fill', 'b');

axis equal;

xlim([0 1]);
ylim([0 1]);

xlabel('Internal vector length', 'FontSize', 8);
ylabel('HD vector length', 'FontSize', 8);


% Panel I - results accuracy plot
axes('position', [grid_3_x grid_2_y * 1.03 x_size1 y_size1]);

colormap jet;
bar(CENTER_OF_HISTOGRAM_BINS, hist_mat', 'stacked');

xlabel('Correlation');
ylabel('Count');

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
[~,start_bin_ind]=min(abs(b-start_value))
[~,end_bin_ind]=min(abs(b-end_value));
%plot(start_value*[1 1],[0 1.1*max(a)],'r-')
%plot(end_value*[1 1],[0 1.1*max(a)],'r-')
%ylim([0 1.1*max(a)])

%figure, plot(b,cumsum(a),'*')

%figure(402)
%plot(log(b),log(cumsum(a)),'r*')
%hold on
% plot([log(b(end))-10 log(b(end))],[log(sum(a))-10 log(sum(a))],'r-')
% plot([log(b(end))-10 log(b(end))],[log(sum(a))-20 log(sum(a))],'r-')

vy=cumsum(a(start_bin_ind:end));
vx=b(start_bin_ind:end)-b(start_bin_ind-1);
relevant_ind=1:(end_bin_ind-start_bin_ind+1);
hold on
plot(log(vx(relevant_ind)),log(vy(relevant_ind)),'r.')
plot(log(vx(relevant_ind(end)+1:end)),log(vy(relevant_ind(end)+1:end)),'r-')
for run1=6:30
    plot([-11 -1],run1+1*[-11 -1],'-','color',slope_one_color)
end
% for run2=8:38
%     plot([-11 -1],run2+2*[-11 -1],'-r')
% end
xlim([-11 -2])
ylim([4 20])

xlabel('Log radius');
ylabel('Log (cum)number of neighbors');


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
% TODO: Magic number at the moment
xlim([0 MAX_FILTRATION_VALUE]);

set(gca, 'xticklabel', []);
set(gca, 'yticklabel', []);

xlabel('Radius');

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

axes('position', [grid_3_x grid_1_y x_size1 y_size1]);
hold on;
xlim([0 MAX_FILTRATION_VALUE]);

set(gca, 'xticklabel', []);
set(gca, 'yticklabel', []);

xlabel('Radius');

axes('position', [grid_4_x grid_1_y x_size1 y_size1]);

scatter(reduced_data_rem_all(:, 2), reduced_data_rem_all(:, 3), 5, cmap2(index_of_visualization_angle_per_temporal_bin_rem_all, :), 'fill');

set(gca, 'xticklabel', []);
set(gca, 'yticklabel', []);

xlabel('Comp. 1');
ylabel('Comp. 2');

colormap gray;
