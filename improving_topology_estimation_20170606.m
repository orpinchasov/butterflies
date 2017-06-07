load('reduced_data_rem_4_6.mat');
persistent_topology;
load('reduced_data_wake_4_6.mat');
persistent_topology;

%%
figure; scatter3(reduced_data(:, 2), reduced_data(:, 3), reduced_data(:, 4), 20, 'k.');

%%
figure; scatter3(centers(:, 1), centers(:, 2), centers(:, 3), 50, 'k.');

%%
figure; scatter3(point_cloud(:, 1), point_cloud(:, 2), point_cloud(:, 3), 50, 'k.');

%%
normalized_distances = distances ./ counts';

figure; histogram(normalized_distances, 50);

%%

[~, sorted_normalized_distances] = sort(normalized_distances);

%%

point_cloud = centers(sorted_normalized_distances(1:50), :);

figure; scatter3(point_cloud(:, 1), point_cloud(:, 2), point_cloud(:, 3), 50, 'k.');

%%
figure; scatter(normalized_distances, counts);

%%

THRESHOLD = 1.25 * 10^-5;

point_cloud = centers(normalized_distances < THRESHOLD, :);

figure; scatter3(point_cloud(:, 1), point_cloud(:, 2), point_cloud(:, 3), 50, 'k.');