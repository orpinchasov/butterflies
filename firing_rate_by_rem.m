load('firing_rate_rem.mat');

figure;
hold on;
% Plot the actual head direction from wake data
plot(filtered_angle_per_temporal_bin, 'k.');

% Plot the estimated head direction using the rem tuning curves and the
% filtered full neuron firing per per, which matches the filtered angle.
smooth_maximum_likelihood_angle_per_sample_index = estimate_head_direction(firing_rate_rem, filtered_full_neuron_firing_per_bin);

plot(smooth_maximum_likelihood_angle_per_sample_index, 'r.');

ylim([0 2 * pi]);
%xlim([2000 6000]);

figure;
scatter(smooth_maximum_likelihood_angle_per_sample_index, filtered_angle_per_temporal_bin', 1, 'k.');
diffs = smooth_maximum_likelihood_angle_per_sample_index - filtered_angle_per_temporal_bin';
histogram(mod(diffs + 1 * pi, 2*pi) - 1 * pi, 30, 'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 1);
xlim([-pi pi]);
%ylim([0 2 * pi]);

figure;
scatter(smooth_maximum_likelihood_angle_per_sample_index, filtered_angle_per_temporal_bin, 1, 'k.');
xlim([0 2 * pi]);
ylim([0 2 * pi]);