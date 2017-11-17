% create_movie(reduced_data, index_of_visualization_angle_per_temporal_bin, cmap2, 'rem');

function create_movie( reduced_data, index_of_angle_per_temporal_bin, cmap, behavioral_state )
%CREATE_MOVIE Summary of this function goes here
%   Detailed explanation goes here

    global NUMBER_OF_ANGLE_BINS;

    hf=figure(100);

    set(gcf,'DefaultAxesFontSize',10);
    set(gcf,'DefaultAxesFontName','arial');
    fig_size_y=15;
    fig_size_x=17;
    set(gcf,'PaperUnits','centimeters','PaperPosition',[5 2 fig_size_x fig_size_y]);
    set(gcf,'PaperOrientation','portrait');
    set(gcf,'Units','centimeters','Position',get(gcf,'paperPosition')+[4 0 0 0]); %+[4 2 0 0]);

    writerObj = VideoWriter('angle_trajectory_on_distribution.avi');
    writerObj.FrameRate = 10;
    open(writerObj);
    set(gcf,'Renderer','zbuffer');

    n=10;
    
    % Main axis
    %if strcmp(behavioral_state, 'wake')
        % Wake and REM with pointer
    %    main_ax = axes('position', [0 0.2 0.8 0.8]);
    %    head_direction_ax = axes('position', [0.8 0.0 0.2 0.2]);
    %else
        main_ax = axes('position', [0 0 1 1]);
    %end

    for run_time=1:1:300
        axes(main_ax);
        hold(main_ax, 'off');
        
        %hold off
        % The third argument is the dot size
        % Color
        scatter(reduced_data(:, 2), reduced_data(:, 3), 5, cmap(index_of_angle_per_temporal_bin,:), 'fill');
        % No color
        %scatter(reduced_data(:, 2), reduced_data(:, 3), 5, 'k', 'fill');
        
        hold(main_ax, 'on');
        
        current_x_vec=reduced_data(run_time+[0:n],2);
        current_y_vec=reduced_data(run_time+[0:n],3);

        current_x=current_x_vec(1);
        current_y=current_y_vec(1);
        
        plot(current_x, current_y, 'ko', 'MarkerFaceColor', [1 1 1], 'MarkerSize', 10, 'LineWidth', 3);
        
        if strcmp(behavioral_state, 'wake')
            xlim([-0.015 0.012]);
            ylim([-0.012 0.015]);
        else
            xlim([-0.019 0.016]);
            ylim([-0.016 0.019]);
        end

        set(main_ax, 'xtick', [-0.01 0 0.01]);
        set(main_ax, 'xticklabel', []);
        set(main_ax, 'ytick', [-0.01 0 0.01]);
        set(main_ax, 'yticklabel', []);
        
        box on;
        
        %xlim(original_xlim);
        %ylim(original_ylim);
        
        %if strcmp(behavioral_state, 'wake')
        if 0
            axes(head_direction_ax);

            hold(head_direction_ax, 'off');
            % Wake
            %indices = circshift(fliplr(1:NUMBER_OF_ANGLE_BINS)', -5);
            % REM
            indices = circshift((1:NUMBER_OF_ANGLE_BINS)', 20);
            scatter(sin([1:NUMBER_OF_ANGLE_BINS]*2*pi/NUMBER_OF_ANGLE_BINS),cos([1:NUMBER_OF_ANGLE_BINS]*2*pi/NUMBER_OF_ANGLE_BINS),20,cmap(indices,:),'fill')
            hold(head_direction_ax, 'on');

            current_angle = pi + index_of_angle_per_temporal_bin(run_time) * 2 * pi / NUMBER_OF_ANGLE_BINS;

            plot([0 sin(current_angle)], [0, cos(current_angle)], 'k-', 'LineWidth', 3);
            
            set(head_direction_ax, 'xtick', []);
            set(head_direction_ax, 'xticklabel', []);
            set(head_direction_ax, 'ytick', []);
            set(head_direction_ax, 'yticklabel', []);
            
            box on;
        
            xlim([-1.2 1.2]);
            ylim([-1.2 1.2]);
        end
        
        drawnow;
        frame = getframe(hf);
        writeVideo(writerObj,frame);
    end

    close(writerObj);
end

