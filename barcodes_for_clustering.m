%% Load hippocampus data

load('HippDataForOr_7p3p17.mat');

%% Filter according to distances between points

d = squareform(pdist(v2(:, 2:4)));

% Eliminate the 0 distance of the diagonal (in order for the min to have a
% meaning)
d2 = d;
d2(d2 == 0) = 1;

% Filter points of v2 according to density factor
v2_dense = v2(min(d2) < 0.0002, :);

% Plot
figure; scatter3(v2_dense(:, 2), v2_dense(:, 3), v2_dense(:, 4), 5, '.');

%% Create barcodes plot and data

data=v2_dense(:, 2:4);
distances = squareform(pdist(data));
distances(distances == 0) = 1; %why?
r_vec = 0.00015:0.00002:0.002;

figure(1);
hold on;

all_bins_mat=nan(length(distances),length(r_vec));
values_array={};
indices_array={};
occurrences_array={};
for i = 2:length(r_vec)
    
    current_graph = distances < r_vec(i);
    
    bins = conncomp(graph(current_graph));
    all_bins_mat(:,i)=bins;
    [values, indices] = unique(bins);
    occurrences = hist(bins, values);
    values_array{i}=values;
    indices_array{i}=indices;
    occurrences_array{i}=occurrences;
    
    values = values(occurrences > 200);
    indices = indices(occurrences > 200);
    
    plot(r_vec(i) * ones(length(values), 1), indices, 'k*');
    
    plot(repmat([r_vec(i) r_vec(i - 1)], [length(values) 1])', repmat(indices, [1 2])', 'k-');
    num_of_clusters(i)=length(values);
end

figure; scatter3(v2_dense(:, 2), v2_dense(:, 3), v2_dense(:, 4), 5, '.');

hold on;
scatter3(v2(bins == 2, 2), v2(bins == 2, 3), v2(bins == 2, 4), 40, 'r.');
figure
plot(r_vec,num_of_clusters,'-')

%% Choose a specific r and bin and analyze

figure(1);

[x y]=ginput(1);
[~,r_index]=min(abs(r_vec-x));
cluster_index=round(y);

figure;
hist(all_bins_mat(:,r_index),[1:140]);

figure; 
scatter3(data(:, 1), data(:, 2), data(:, 3), 5, '.');
hold on;
chosen_points=~~(all_bins_mat(:,r_index)==all_bins_mat(cluster_index,r_index));
scatter3(data(chosen_points, 1), data(chosen_points, 2), data(chosen_points, 3), 40, 'r.');

%% Create fixed indexing arrays by keeping the cluster number of the larger
% cluster when merging

new_ind_array={};
new_ind_array{2} = indices_array{2};
for i = 3:length(r_vec)
    current_indices = indices_array{i};
    current_bins = all_bins_mat(:, i);
    prev_indices = indices_array{i - 1};
    prev_bins = all_bins_mat(:, i - 1);
    prev_occurr = occurrences_array{i - 1};
    
    prev_new_ind = new_ind_array{i - 1};
    
    new_ind_vec = nan(1, length(current_indices));
    for run_cluster=1:length(current_indices)
        sons_of_run_clusters = prev_new_ind(current_bins(prev_indices) == prev_bins(prev_indices(run_cluster)));
        occurr_of_sons = prev_occurr(current_bins(prev_indices) == prev_bins(prev_new_ind(run_cluster)));
        [~, most_freq_son_ind] = max(occurr_of_sons);

        new_ind_vec(run_cluster) = prev_new_ind(prev_new_ind == sons_of_run_clusters(most_freq_son_ind));
            
    end
    
    new_ind_array{i} = new_ind_vec; 
end

%% Plot fixed indices barcodes

figure(1);
hold on;

for i = 2:length(r_vec)
    indices = new_ind_array{i};
    occurrences = occurrences_array{i};
    
    indices = indices(occurrences > 100);
    
    plot(r_vec(i) * ones(length(indices), 1), indices, 'k*');
end

%% Plot a grayscale map of fixed indices using intensity as a marker for number
% of data points.

gray_map=[];
for i = 3:length(r_vec)
    gray_map(new_ind_array{i},i)=occurrences_array{i};
end

gray_map_reduced=gray_map(:,30:end);
gray_map_reduced=gray_map(sum(gray_map_reduced,2)>0,:);
figure;
imagesc(-gray_map_reduced);
colormap gray;

%% Plot occurrences histogram for occurrences

huge_occurr_vec=[];
for i = 2:length(r_vec)
    huge_occurr_vec=[huge_occurr_vec occurrences_array{i}];
end

figure;
hist(huge_occurr_vec,1000);

%% Choose multiple barcodes and draw the clusters they impose

figure(101);
scatter3(data(:, 1), data(:, 2), data(:, 3), 5, '.');
hold on;

n=0;
color_map=lines(10);
while 1
    n=n+1;
    figure(1);

    [x y]=ginput(1);
    [~,r_index]=min(abs(r_vec-x));
    cluster_index=round(y);
    hold on;
    plot(x,y,'*','color',color_map(n,:));

    figure(101);
    chosen_points=~~(all_bins_mat(:,r_index)==all_bins_mat(cluster_index,r_index));
    plot3(data(chosen_points, 1), data(chosen_points, 2), data(chosen_points, 3), '.','color',color_map(n,:),'MarkerSize',5);
end