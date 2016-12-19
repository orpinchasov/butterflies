function plot_angle_and_states( Ang, wake, rem, sws )
    global BEHAVIORAL_SAMPLE_RATE;

    LINE_WIDTH = 2;

    figure;
    hold on;
    plot(Ang, '.');

    for i = 1:size(rem, 1);
        plot([1 1] * rem(i, 1) * BEHAVIORAL_SAMPLE_RATE, [0 2 * pi], 'r-', 'LineWidth', LINE_WIDTH);
        plot([1 1] * rem(i, 2) * BEHAVIORAL_SAMPLE_RATE, [0 2 * pi], 'r-', 'LineWidth', LINE_WIDTH);

        plot(rem(i, 1:2) * BEHAVIORAL_SAMPLE_RATE, [0 0], 'r-', 'LineWidth', LINE_WIDTH);
        plot(rem(i, 1:2) * BEHAVIORAL_SAMPLE_RATE, [2 * pi 2 * pi], 'r-', 'LineWidth', LINE_WIDTH);
    end

    for i = 1:size(sws, 1);
        plot([1 1] * sws(i, 1) * BEHAVIORAL_SAMPLE_RATE, [0 2 * pi], 'b-', 'LineWidth', LINE_WIDTH);
        plot([1 1] * sws(i, 2) * BEHAVIORAL_SAMPLE_RATE, [0 2 * pi], 'b-', 'LineWidth', LINE_WIDTH);

        plot(sws(i, 1:2) * BEHAVIORAL_SAMPLE_RATE, [0 0], 'b-', 'LineWidth', LINE_WIDTH);
        plot(sws(i, 1:2) * BEHAVIORAL_SAMPLE_RATE, [2 * pi 2 * pi], 'b-', 'LineWidth', LINE_WIDTH);
    end

    for i = 1:size(wake, 1);
        plot([1 1] * wake(i, 1) * BEHAVIORAL_SAMPLE_RATE, [0 2 * pi], 'g-', 'LineWidth', LINE_WIDTH);
        plot([1 1] * wake(i, 2) * BEHAVIORAL_SAMPLE_RATE, [0 2 * pi], 'g-', 'LineWidth', LINE_WIDTH);

        plot(wake(i, 1:2) * BEHAVIORAL_SAMPLE_RATE, [0 0], 'g-', 'LineWidth', LINE_WIDTH);
        plot(wake(i, 1:2) * BEHAVIORAL_SAMPLE_RATE, [2 * pi 2 * pi], 'g-', 'LineWidth', LINE_WIDTH);
    end
    
    figure;
    plot(Ang(round(wake(1) * BEHAVIORAL_SAMPLE_RATE):round(wake(2) * BEHAVIORAL_SAMPLE_RATE)), '.');
end

