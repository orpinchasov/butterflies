sigma = 2;
hsize = 5;
gaussian_kernel=exp(-(([1:hsize]-(hsize+1)/2).^2)/(2*sigma^2));
gaussian_kernel=gaussian_kernel./sum(gaussian_kernel);

try
    load('angle_per_temporal_bin')
    load('estimated_angle_by_clustering')
catch ME
    ME
end

smoothed_estimated_angle_by_clustering=angle(conv2(exp(1i*estimated_angle_by_clustering),gaussian_kernel,'same'));
smoothed_estimated_angle_by_clustering=mod(smoothed_estimated_angle_by_clustering,2*pi);

%%
figure
hold on
plot(angle_per_temporal_bin,'k.')
plot(estimated_angle_by_clustering,'r.')
xlim([0.2 0.7] * 10^4);
ylim([0 2 * pi]);

%%
figure
hold on
plot([1:size(angle_per_temporal_bin, 1)] / (BEHAVIORAL_SAMPLE_RATE / BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN), angle_per_temporal_bin,'k.')
plot([1:size(smoothed_estimated_angle_by_clustering, 2)] / (BEHAVIORAL_SAMPLE_RATE / BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN), smoothed_estimated_angle_by_clustering,'r.')
xlim([0.2 0.7] * 10^4 / (BEHAVIORAL_SAMPLE_RATE / BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN));
xlabel('Time (sec)');
ylim([0 2 * pi]);

%%
figure
subplot(1,2,1)
plot(angle_per_temporal_bin,estimated_angle_by_clustering,'k.')
subplot(1,2,2)
plot(angle_per_temporal_bin,smoothed_estimated_angle_by_clustering,'k.')