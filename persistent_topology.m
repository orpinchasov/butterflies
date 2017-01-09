%% Parameters
% Clustering
NUMBER_OF_CLUSTERS = 50;
NUMBER_OF_SAMPLES_PER_CLUSTER = 10;
TOTAL_NUMBER_OF_SAMPLES = 50;

% Topology calculation
MAX_DIMENSION = 3;
MAX_FILTRATION_VALUE = 0.015;
%MAX_FILTRATION_VALUE = 0.9;
NUM_DIVISIONS = 20;

%% Cluster data and choose a small number of points to pass to the topology alogithm
all_indices = zeros(NUMBER_OF_SAMPLES_PER_CLUSTER * NUMBER_OF_CLUSTERS, 1);

labels = k_means_clustering(reduced_data, NUMBER_OF_CLUSTERS, 1);

for i = 1:NUMBER_OF_CLUSTERS
    cluster_indices = find(labels == i);
    
    random_indices = randsample(cluster_indices, NUMBER_OF_SAMPLES_PER_CLUSTER);
    
    all_indices(((i - 1) * NUMBER_OF_SAMPLES_PER_CLUSTER + 1):i * NUMBER_OF_SAMPLES_PER_CLUSTER) = random_indices;
end

%% Cluster data and get the average point of each cluster
average_cluster_data_points = zeros(NUMBER_OF_CLUSTERS, 1);

rng(0);
[labels centers] = kmeans(reduced_data(:, 2:4), NUMBER_OF_CLUSTERS);

counts = histcounts(labels, 0.5:1:NUMBER_OF_CLUSTERS + 0.5);

%% Find the distances between the points and remove those which are too close.
% Continue until reaching the required number of data points.

all_indices_filtered = all_indices;

% Remove some of the closest points in each cluster in order to get to the
% required number of data points.
for i = 1:NUMBER_OF_CLUSTERS
    cluster_indices = all_indices(((i - 1) * NUMBER_OF_SAMPLES_PER_CLUSTER + 1):i * NUMBER_OF_SAMPLES_PER_CLUSTER);
    
    distances = pdist(reduced_data(cluster_indices, 2:3));
    distances = squareform(distances);

    distances(distances == 0) = inf;
    
    % TODO: The range is not correct here
    for j = 1:round(length(all_indices) / NUMBER_OF_SAMPLES_PER_CLUSTER)
        [row, col] = find(distances == min(distances(:)));
        
        distances(row, col) = inf;
        
        % Remove one of the two neurons
        original_value = cluster_indices(row);
        cluster_indices(row) = nan;
        all_indices_filtered(find(all_indices_filtered == original_value)) = nan;
    end
    
    all_indices_filtered
end

%% Get data ready

%point_cloud = reduced_data(all_indices, 2:3);
point_cloud = centers(~(counts < 50), :);
%point_cloud = pointsTorusGrid;

figure;
hold on;
%scatter(reduced_data(:, 2), reduced_data(:, 3), '.');
scatter(point_cloud(:, 1), point_cloud(:, 2), '.r');
%scatter(point_cloud(counts < 50, 1), point_cloud(counts < 50, 2), '.k');

%% Create
'Create'
stream = api.Plex4.createVietorisRipsStream(point_cloud, MAX_DIMENSION, MAX_FILTRATION_VALUE, NUM_DIVISIONS);
'Done'

%% ...
'...'
persistence = api.Plex4.getModularSimplicialAlgorithm(MAX_DIMENSION, 2);
intervals = persistence.computeIntervals(stream);
'Done'

%% Plot output
options.filename = 'ReducedDataHeadDirection';
options.max_filtration_value = MAX_FILTRATION_VALUE;
options.max_dimension = MAX_DIMENSION - 1;
%options.max_dimension = 1;
options.side_by_side = true;
handles = plot_barcodes(intervals, options);

%% Plot all clusters and connect each two by a line if the distance between
% them is smaller than 0.0075 (somewhere in the range of the previous
% result).
figure;
hold on;

scatter(centers(:, 1), centers(:, 2), 40, '.r');

distances = pdist(centers);
distances = squareform(distances);
%distances(distances == 0) = inf;

[row, col] = find(distances < 0.0015);

%line([centers(row, 1) centers(col, 1)], [centers(row, 2) centers(col, 2)]);

for i = 1:NUMBER_OF_CLUSTERS
    for j = 1:NUMBER_OF_CLUSTERS
        if distances(i, j) < 0.006
            plot([centers(i, 1) centers(j, 1)], [centers(i, 2) centers(j, 2)], 'k-');
        end
    end
end

%% Get topology data
dimension_0 = homology.barcodes.BarcodeUtility.getEndpoints(intervals, 0, false);
dimension_1 = homology.barcodes.BarcodeUtility.getEndpoints(intervals, 1, false);

figure;
subplot(1, 3, 1);
hold on;
number_of_results = size(dimension_0, 1);
ylim([0 number_of_results + 1]);
for i = 1:number_of_results
    if dimension_0(i, 2) == Inf
        plot([dimension_0(i, 1) 0.015], [number_of_results + 1 - i number_of_results + 1 - i], 'b');
    else
        plot([dimension_0(i, 1) dimension_0(i, 2)], [number_of_results + 1 - i number_of_results + 1 - i], 'b');
    end
end
% TODO: Magic number at the moment
xlim([0 0.015]);

subplot(1, 3, 2);
hold on;
number_of_results = size(dimension_1, 1);
ylim([0 number_of_results + 1]);
for i = 1:number_of_results
    if dimension_1(i, 2) == Inf
        plot([dimension_1(i, 1) 0.015], [number_of_results + 1 - i number_of_results + 1 - i], 'b');
    else
        plot([dimension_1(i, 1) dimension_1(i, 2)], [number_of_results + 1 - i number_of_results + 1 - i], 'b');
    end
end
xlim([0 0.015]);

subplot(1, 3, 3);
hold on;
xlim([0 0.015]);