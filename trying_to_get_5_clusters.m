%%
cd('C:\Users\orp\Downloads\matlab_examples');
load_javaplex;
cd('d:\dev\alon_data\code\custom_scripts');

load('HippDataForOr_7p3p17.mat');

%% Parameters
% Clustering
NUMBER_OF_TOPOLOGY_CLUSTERS = 200;
NUMBER_OF_SAMPLES_PER_CLUSTER = 10;
TOTAL_NUMBER_OF_SAMPLES = 50;
CLUSTERING_DIMENSIONS = 2:4;

% Topology calculation
MAX_DIMENSION = 3;
MAX_FILTRATION_VALUE = 0.01;
%MAX_FILTRATION_VALUE = 0.9;
NUM_DIVISIONS = 100;

%%
addpath('D:\dev\speclust1_0');

option.similar='eucl'; % used by constructSimMatrix
option.sigma=2^6; % used by constructSimMatrix
option.graph='knn'; % used by constructSimGraph, only 'knn' is implemented
option.kneighbor=10; % used by constructSimGraph
option.normalizedLaplacian=1; % set to 1
option.clusterMethod='kmean'; % used by clusters, so far, you can only use k-mean, but you can also use NMF (see my NMF MATLAB Toolbox) 
k=NUMBER_OF_TOPOLOGY_CLUSTERS; % number of clusters
% spectral clustering
[labels,clusterValids, centers]=spectralCluster(v2(1:3000, CLUSTERING_DIMENSIONS)',k,option);

centers = centers';

counts = histcounts(labels, 0.5:1:NUMBER_OF_TOPOLOGY_CLUSTERS + 0.5);

%% K-Means clustering
rng(0);
[labels centers] = kmeans(v2(:, CLUSTERING_DIMENSIONS), NUMBER_OF_TOPOLOGY_CLUSTERS);

counts = histcounts(labels, 0.5:1:NUMBER_OF_TOPOLOGY_CLUSTERS + 0.5);

%% Spectral clustering
rng(0);

M = v2(:, CLUSTERING_DIMENSIONS);
W = squareform(pdist(M));
sigma = 0.05;
similarity_spectral_clustering_distances = exp(-W.^2 ./ (2*sigma^2));

[labels, centers] = SpectralClustering(similarity_spectral_clustering_distances, NUMBER_OF_TOPOLOGY_CLUSTERS, 3);

labels = log2(bi2de(labels)) + 1;

counts = histcounts(labels, 0.5:1:NUMBER_OF_TOPOLOGY_CLUSTERS + 0.5);

%% Get data ready

%point_cloud = reduced_data(all_indices, 2:3);
centers = centers(~(counts < 50), :);
%point_cloud = pointsTorusGrid;

%% 

few_points = v2(1:3000, 2:4);

max_distances = zeros(NUMBER_OF_TOPOLOGY_CLUSTERS, 1);
for i = 1:NUMBER_OF_TOPOLOGY_CLUSTERS
    max_distances(i) = std(pdist(few_points(labels == i)));
end

%%
r = 1:NUMBER_OF_TOPOLOGY_CLUSTERS;

zero_values = r(max_distances < 0.0005);
zero_values_size = r(counts < 20);

zero_values = intersect(zero_values, zero_values_size);

v_labels = labels;

reduced_points = ismember(v_labels, zero_values);

%for i = 1:length(zero_values)
%    v_labels(v_labels == zero_values(i)) = 1;
%end

figure; scatter3(few_points(:, 1), few_points(:, 2), few_points(:, 3), 20, clustering_cmap(v_labels, :), '.');
figure; scatter3(few_points(reduced_points, 1), few_points(reduced_points, 2), few_points(reduced_points, 3), 20, clustering_cmap(v_labels(reduced_points), :), '.');


%% Plot data points

clustering_cmap = hsv(NUMBER_OF_TOPOLOGY_CLUSTERS);

figure; scatter3(v2(:, 2), v2(:, 3), v2(:, 4), 20, '.');

figure; scatter3(v2(1:3000, 2), v2(1:3000, 3), v2(1:3000, 4), 20, clustering_cmap(labels, :), '.');

figure; scatter3(centers(:, 1), centers(:, 2), centers(:, 3));

%% Create
'Create'
stream = api.Plex4.createVietorisRipsStream(centers, MAX_DIMENSION, MAX_FILTRATION_VALUE, NUM_DIVISIONS);
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


radius_expansion_steps = [0 0.0035 0.007 0.0105 0.014];

centers = point_cloud;

distances = pdist(centers);
distances = squareform(distances);

for radius_expansion_step = radius_expansion_steps
    figure;
    hold on;

    A=zeros(size(distances));
    A(distances<radius_expansion_step)=1;
    A_sqaured=A*A;

    clicks=zeros(3,100000);
    num_clicks=0;
    for n=1:size(A,1)
        for m=n+1:size(A,2)
            if A(n,m)>0 & A_sqaured(n,m)>0
                num_new_clicks=A_sqaured(n,m)-2;
                num_clicks=num_clicks+num_new_clicks;
                A_two_rows_temp=A([n m],:);
                A_two_rows_temp(:,[n m])=[0 0 ; 0 0];
                temp_third_members=find(sum(A_two_rows_temp)==2);
                clicks(:,num_clicks-num_new_clicks+1:num_clicks)=[n*ones(1,num_new_clicks) ; m*ones(1,num_new_clicks) ; temp_third_members];
            end
        end
    end

    x_positions = centers(:, 1);
    y_positions = centers(:, 2);

    clicks(:,num_clicks+1:end)=[];
    for n=1:num_clicks
        x_locs=x_positions(clicks(:,n));
        y_locs=y_positions(clicks(:,n));
        p=patch(x_locs,y_locs,[1 1 0]);
        set(p,'FaceAlpha',0.2,'EdgeColor','none');
    end

    for i = 1:size(centers, 1)
        for j = 1:size(centers, 1)
            if distances(i, j) < radius_expansion_step
                plot([centers(i, 1) centers(j, 1)], [centers(i, 2) centers(j, 2)], 'k-');
            end
        end
    end

    scatter(centers(:, 1), centers(:, 2), 300, '.r');

end
