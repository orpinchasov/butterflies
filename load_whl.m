constants

global BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN;

%%
FILENAME = 'E:\or\data\Mouse28-140312\Mouse28-140312.whl';

Fp = fopen(FILENAME, 'r');

whl = fscanf(Fp, '%f %f %f %f');
fclose(Fp);
%%
whl = reshape(whl, [4 length(whl) / 4])';

whl(whl == -1) = nan;

%%
x1 = whl(:, 1);
y1 = whl(:, 2);
x2 = whl(:, 3);
y2 = whl(:, 4);

%% Extract samples of wake period
wake_x1 = x1(ceil(wake(1) * BEHAVIORAL_SAMPLE_RATE):floor(wake(2) * BEHAVIORAL_SAMPLE_RATE));
wake_y1 = y1(ceil(wake(1) * BEHAVIORAL_SAMPLE_RATE):floor(wake(2) * BEHAVIORAL_SAMPLE_RATE));
wake_x2 = x2(ceil(wake(1) * BEHAVIORAL_SAMPLE_RATE):floor(wake(2) * BEHAVIORAL_SAMPLE_RATE));
wake_y2 = y2(ceil(wake(1) * BEHAVIORAL_SAMPLE_RATE):floor(wake(2) * BEHAVIORAL_SAMPLE_RATE));

wake_ang = Ang(ceil(wake(1) * BEHAVIORAL_SAMPLE_RATE):floor(wake(2) * BEHAVIORAL_SAMPLE_RATE));
        
%% Truncate each vector to nearest multiplication of 4
wake_x1 = wake_x1(1:end - mod(length(wake_x1), 4));
wake_y1 = wake_y1(1:end - mod(length(wake_y1), 4));
wake_x2 = wake_x2(1:end - mod(length(wake_x2), 4));
wake_y2 = wake_y2(1:end - mod(length(wake_y2), 4));

wake_ang = wake_ang(1:end - mod(length(wake_ang), 4));

%% Average data
% TODO: Change 4 to BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN where applicable
averaged_wake_x1 = nanmean(reshape(wake_x1, [BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN size(wake_x1, 1) / 4]));
averaged_wake_y1 = nanmean(reshape(wake_y1, [BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN size(wake_y1, 1) / 4]));
averaged_wake_x2 = nanmean(reshape(wake_x2, [BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN size(wake_x2, 1) / 4]));
averaged_wake_y2 = nanmean(reshape(wake_y2, [BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN size(wake_y2, 1) / 4]));

wake_ang = reshape(wake_ang, [BEHAVIORAL_SAMPLES_PER_TEMPORAL_BIN length(wake_ang) / 4])';

%%
angles_valid_mask_mat = (wake_ang ~= -1);
angle_per_temporal_bin = mod(angle(sum(angles_valid_mask_mat .* exp(1i * wake_ang), 2)), 2 * pi);
angle_per_temporal_bin(abs(sum(angles_valid_mask_mat .* exp(1i * wake_ang), 2)) == 0) = nan;
angles = angle_per_temporal_bin';

%%
x_diff = averaged_wake_x2 - averaged_wake_x1;
y_diff = averaged_wake_y2 - averaged_wake_y1;

x_vec = 0.5 * (averaged_wake_x2 + averaged_wake_x1);
y_vec = 0.5 * (averaged_wake_y2 + averaged_wake_y1);

x_velocity = diff(x_vec);
y_velocity = diff(y_vec);

led_angle = angle(x_diff + i * y_diff);

velocity_angle = angle(x_velocity + i * y_velocity);

figure;
plot(led_angle, angles, '.');

figure;
plot(led_angle(1:end - 1), velocity_angle, '.');

%% Calculate the distance between the two LEDs across time. This should enable
% us to estimate the angle of the head of the mouse (up or down).
inter_led_interval = sqrt(x_diff.^2 + y_diff.^2);
figure;
plot(inter_led_interval, '.');

%% Calculate angular velocity and correct for velocities close to 0 or 2 * pi
figure;
corrected_diff = mod(diff(angles) + pi, 2 * pi) - pi;
plot(corrected_diff, '.');

%% Bin angular velocity
angular_velocity = corrected_diff;
