sigma = 2;
hsize = 5;
gaussian_kernel=exp(-(([1:hsize]-(hsize+1)/2).^2)/(2*sigma^2));
gaussian_kernel=gaussian_kernel./sum(gaussian_kernel);

load('angle_per_temporal_bin')
load('estimated_angle_by_clustering')

smoothed_estimated_angle_by_clustering=angle(conv2(exp(1i*estimated_angle_by_clustering),gaussian_kernel,'same'));
smoothed_estimated_angle_by_clustering=mod(smoothed_estimated_angle_by_clustering,2*pi);

figure
hold on
plot(angle_per_temporal_bin,'k.')
plot(estimated_angle_by_clustering,'r.')

figure
hold on
plot(angle_per_temporal_bin,'k.')
plot(smoothed_estimated_angle_by_clustering,'r.')

figure
subplot(1,2,1)
plot(angle_per_temporal_bin,estimated_angle_by_clustering,'k.')
subplot(1,2,2)
plot(angle_per_temporal_bin,smoothed_estimated_angle_by_clustering,'k.')