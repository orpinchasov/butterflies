function [ reduced_data ] = create_reduced_data( data, p_neightbors_vec, number_of_reduced_dimensions_vec )
%CREATE_REDUCED_DATA Summary of this function goes here
%   Detailed explanation goes here

    reduced_data = data;
    
    for iteration_index = 1:length(number_of_reduced_dimensions_vec)
        reduced_data = reduce_data_by_laplacian_eigenmap(reduced_data, ...
                                                         p_neightbors_vec(iteration_index), ...
                                                         number_of_reduced_dimensions_vec(iteration_index));
    end
end

