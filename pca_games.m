%function reduced_data = reduce_data_by_laplacian_eigenmap(data, p_neighbors, number_of_reduced_dimensions)
data = ordered_transition_mat;
number_of_reduced_dimensions = 5;

% N = size(data, 1);
% 
% % Changing these values will lead to different nonlinear embeddings
% %knn = ceil(p_neighbors*N); % each patch will only look at its knn nearest neighbors in R^d
% knn = 2; % Equivalent to p_neighbors = 0.25
% 
% % now let's get pairwise distance info and create graph 
% m             = N;
% distances_mat = ordered_transition_mat; % Create NxN matrix of the distance
% [~, srtdIdx]  = sort(distances_mat, 'ascend');
% nidx          = srtdIdx(1:knn + 1,:);
% 
% % Weights matrix
% tempW = ones(size(nidx));
% 
% % Build weight matrix
% i = repmat(1:m, knn + 1, 1);
% % i - sample number, nidx - sample number of nearest neighbor,
% % tempW - weight of connectivity between near neighbors (in our case
% % always 1).
% W = sparse(i(:), double(nidx(:)), tempW(:), m, m);
% W = max(W, W'); % for undirected graph.

%W = ordered_transition_mat;
%W = 0.5*(ordered_transition_mat+ordered_transition_mat');
W = max(ordered_transition_mat,ordered_transition_mat');


% The original normalized graph Laplacian, non-corrected for density
ld = diag(sum(W,2).^(-1/2));
DO = ld*W*ld;
DO = max(DO,DO');%(DO + DO')/2;
%DO = diag(sum(W,2))/length(W)-W;

% get eigenvectors
[v,d] = eigs(DO,number_of_reduced_dimensions,'la');

%% 'v' also known as 'transition_matrix_states'
figure;

scatter(-v(:, 2), v(:, 3), 300, cmap_clusters, 'fill');

axis equal;
box;

xlim([-0.7 0.7]);
ylim([-0.8 0.6]);