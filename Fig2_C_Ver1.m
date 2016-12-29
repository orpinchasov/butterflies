%function FigSup4_AB_VER0(x_pos1,y_pos1,x_pos2,y_pos2,x_size1,y_size1,x_size2,y_size2)

cage_num_array={'13' '12' '11' '7' '6'};
mouse_num_array={'1' '4' '1' '4' '4'};
Ver_num_array={'5' '2' '1' '2' '1'};
%Ver_num_array={'4' '3' '2' '5' '2'};
day1_start_with_B_track_vec=[1 1 1 1 0];
Nbins=100;
num_of_active_days_threshold=-1;
shuffle_type2_flag=1;
shuffled_p1_values=[];
shuffled_p2_values=[];
shuffled_p3_values=[];

if 1
    n=8;
% current_corr_coeff_vec=nan(1,factorial(n));
% comlete_shuffle_vec=nan(1,factorial(n));
shuffle_mat=nan(factorial(n),n);
for run_shuffle=1:factorial(n)
    temp_shuffle_vec=nan(1,n);
    index_n=run_shuffle;
    for run_shuffle_index=n:-1:2
        shuffle_index_digit=ceil(index_n/factorial(run_shuffle_index-1));
        index_n=index_n-(shuffle_index_digit-1)*factorial(run_shuffle_index-1);
        temp_shuffle_vec(run_shuffle_index)=shuffle_index_digit;
    end
    temp_shuffle_vec(1)=1;
    shuffle_vec=nan(1,n);
    for run_shuffle_index=n:-1:1
        nan_vec=nan(1,run_shuffle_index);
        nan_vec(temp_shuffle_vec(run_shuffle_index))=run_shuffle_index;
        shuffle_vec(isnan(shuffle_vec))=nan_vec;
    end
    shuffle_mat(run_shuffle,:)=shuffle_vec;
end
ind_mat=sub2ind([8 8], shuffle_mat(:,1:7)', shuffle_mat(:,2:8)');
end
toc

p_A_vec=[];
p_B_vec=[];
p_all_active_vec=[];
for run_mice=1:5
    run_mice
    cage_num=cage_num_array{run_mice};
    mouse_num=mouse_num_array{run_mice};
    Ver_num=Ver_num_array{run_mice};
    day1_start_with_B_track=day1_start_with_B_track_vec(run_mice);
    
    [number_of_events_matrix_cell_by_session_by_trial number_of_events_matrix_cell_by_trial_A_session number_of_events_matrix_cell_by_trial_B_session number_of_events_matrix_cell_by_session_A_session number_of_events_matrix_cell_by_session_B_session]=...
        built_num_of_events_mat_VER0(cage_num,mouse_num,Ver_num,day1_start_with_B_track);
    
    
    num_of_days_each_cell_is_active=sum(~~(number_of_events_matrix_cell_by_session_A_session+number_of_events_matrix_cell_by_session_B_session),2);
    cells_inclusion_list=(num_of_days_each_cell_is_active>num_of_active_days_threshold);
    
    number_of_events_matrix_cell_by_day=number_of_events_matrix_cell_by_session_A_session;
    number_of_events_matrix_cell_by_day=number_of_events_matrix_cell_by_day(cells_inclusion_list,:);
    corr_mat_activity_all_cells=corr(number_of_events_matrix_cell_by_day);
    shuffled_activity_all_cells_values=mean(corr_mat_activity_all_cells(ind_mat));
    %[hist_shuffled_values bins_centers]=hist(shuffled_activity_all_cells_values,linspace(-1,6,60));
    %subplot(3,1,1)
    %bar(bins_centers,hist_shuffled_values/2)
    %hold on
    %max_y=250*ceil(max(hist_shuffled_values)/500)+100;
    %plot(shuffled_activity_all_cells_values(1)*[1 1],[0 max_y],'r-')
    n1=sum(shuffled_activity_all_cells_values(1)<shuffled_activity_all_cells_values(2:end-1))
    p1=n1/length(shuffled_activity_all_cells_values(2:end-1))
    %title(['p=' num2str(p1) ', n=' num2str(n1/2)])
    
    number_of_events_matrix_cell_by_day=number_of_events_matrix_cell_by_session_B_session;
    number_of_events_matrix_cell_by_day=number_of_events_matrix_cell_by_day(cells_inclusion_list,:);
    corr_mat_activity_all_cells=corr(number_of_events_matrix_cell_by_day);
    shuffled_activity_all_cells_values=mean(corr_mat_activity_all_cells(ind_mat));
    %[hist_shuffled_values bins_centers]=hist(shuffled_activity_all_cells_values,linspace(-1,6,60));
    %subplot(3,1,2)
    %bar(bins_centers,hist_shuffled_values/2)
    %hold on
    %max_y=250*ceil(max(hist_shuffled_values)/500)+100;
    %plot(shuffled_activity_all_cells_values(1)*[1 1],[0 max_y],'r-')
    n2=sum(shuffled_activity_all_cells_values(1)<shuffled_activity_all_cells_values(2:end-1))
    p2=n2/length(shuffled_activity_all_cells_values(2:end-1))
    %title(['p=' num2str(p2) ', n=' num2str(n2/2)])
    
    number_of_events_matrix_cell_by_day=number_of_events_matrix_cell_by_session_A_session+number_of_events_matrix_cell_by_session_B_session;
    number_of_events_matrix_cell_by_day=number_of_events_matrix_cell_by_day(cells_inclusion_list,:);
    corr_mat_activity_all_cells=corr(number_of_events_matrix_cell_by_day);
    shuffled_activity_all_cells_values=mean(corr_mat_activity_all_cells(ind_mat));
    [hist_shuffled_values bins_centers]=hist(shuffled_activity_all_cells_values,linspace(0,1,100));
%     %subplot(3,1,3)
%     bar(bins_centers,hist_shuffled_values/factorial(8))
%     hold on
%     %max_y=250*ceil(max(hist_shuffled_values)/500)+100;
%     max_y_for_hist=ceil(max(hist_shuffled_values/factorial(8))*10)/10;
%     plot(shuffled_activity_all_cells_values(1)*[1 1],[0 max_y_for_hist],'r-')
%     xlim([0 1])
%     ylim([0 max_y_for_hist])
    n3=sum(shuffled_activity_all_cells_values(1)<shuffled_activity_all_cells_values(2:end-1))
    p3=n3/length(shuffled_activity_all_cells_values(2:end-1))
    %title(['p=' num2str(p3) ', n=' num2str(n3/2)])
%     title(['Mouse ' num2str(run_mice) ])
    
    p_A_vec=[p_A_vec p1];
    p_B_vec=[p_B_vec p2];
    p_all_active_vec=[p_all_active_vec p3];
    
    if shuffle_type2_flag
        rand('seed',14)
        for run_shuffles_type2=1:10
        for  run_cells=1:size(number_of_events_matrix_cell_by_session_A_session,1);
            number_of_events_matrix_cell_by_session_A_session(run_cells,:)=number_of_events_matrix_cell_by_session_A_session(run_cells,randperm(8));
            number_of_events_matrix_cell_by_session_B_session(run_cells,:)=number_of_events_matrix_cell_by_session_B_session(run_cells,randperm(8));
        end
        num_of_days_each_cell_is_active=sum(~~(number_of_events_matrix_cell_by_session_A_session+number_of_events_matrix_cell_by_session_B_session),2);
        cells_inclusion_list=(num_of_days_each_cell_is_active>num_of_active_days_threshold);
        
        number_of_events_matrix_cell_by_day=number_of_events_matrix_cell_by_session_A_session;
        number_of_events_matrix_cell_by_day=number_of_events_matrix_cell_by_day(cells_inclusion_list,:);
        corr_mat_activity_all_cells=corr(number_of_events_matrix_cell_by_day);
        shuffled_activity_all_cells_values=mean(corr_mat_activity_all_cells(ind_mat));
        n1=sum(shuffled_activity_all_cells_values(1)<shuffled_activity_all_cells_values(2:end-1));
        p1=n1/length(shuffled_activity_all_cells_values(2:end-1));
        
        number_of_events_matrix_cell_by_day=number_of_events_matrix_cell_by_session_B_session;
        number_of_events_matrix_cell_by_day=number_of_events_matrix_cell_by_day(cells_inclusion_list,:);
        corr_mat_activity_all_cells=corr(number_of_events_matrix_cell_by_day);
        shuffled_activity_all_cells_values=mean(corr_mat_activity_all_cells(ind_mat));
        n2=sum(shuffled_activity_all_cells_values(1)<shuffled_activity_all_cells_values(2:end-1));
        p2=n2/length(shuffled_activity_all_cells_values(2:end-1));
        
        number_of_events_matrix_cell_by_day=number_of_events_matrix_cell_by_session_A_session+number_of_events_matrix_cell_by_session_B_session;
        number_of_events_matrix_cell_by_day=number_of_events_matrix_cell_by_day(cells_inclusion_list,:);
        corr_mat_activity_all_cells=corr(number_of_events_matrix_cell_by_day);
        shuffled_activity_all_cells_values=mean(corr_mat_activity_all_cells(ind_mat));
        n3=sum(shuffled_activity_all_cells_values(1)<shuffled_activity_all_cells_values(2:end-1));
        p3=n3/length(shuffled_activity_all_cells_values(2:end-1));
        
        shuffled_p1_values(run_mice,run_shuffles_type2)=p1;
        shuffled_p2_values(run_mice,run_shuffles_type2)=p2;
        shuffled_p3_values(run_mice,run_shuffles_type2)=p3;
        end
    end
end
% 
% figure;
% plot(log10(p_A_vec+2/factorial(8)),log10(p_B_vec+2/factorial(8)),'*')
% xlim([log10(2/factorial(8))-10*eps log10(1)])
% ylim([log10(2/factorial(8))-10*eps log10(1)])
% hold on
% plot([log10(2/factorial(8)) log10(1)],[log10(0.05) log10(0.05)] ,'r-')
% plot([log10(0.05) log10(0.05)] ,[log10(2/factorial(8)) log10(1)],'r-')
% plot(log10(2/factorial(8)),log10(2/factorial(8)),'ro')
% xlabel('log10(p) linear session')
% ylabel('log10(p) L-shape session')


% mice_color_mat=[0.8 0.6 0;0 1 0;0 0.65 0.75; 0 0 1; 1 0 1];
% for run_mice=1:5
%     color_of_current_mouse=mice_color_mat(run_mice,:);
%     plot(log(p_A_vec(run_mice)+2/factorial(8)),log(p_B_vec(run_mice)+2/factorial(8)),'o','color',color_of_current_mouse,'MarkerFaceColor',color_of_current_mouse)
%     hold on
% end
plot(log(p_A_vec+2/factorial(8)),log(p_B_vec+2/factorial(8)),'o','color',[0 0.5 1],'MarkerFaceColor',[0 0.5 1])
hold on

xlim([log(2/factorial(8))-0*eps log(1)])
ylim([log(2/factorial(8))-0*eps log(1)])
plot(log(2/factorial(8)),log(2/factorial(8)),'o','color',[1 0 0],'MarkerFaceColor',[1 0 0])
% xlabel('log(p) linear session')
% ylabel('log(p) L-shape session')

plot(log(shuffled_p1_values(:)),log(shuffled_p2_values(:)),'o','color',[0.35 0.35 0.35],'MarkerFaceColor',[0.4 0.4 0.4])
plot([log(2/factorial(8)) log(1)],[log(0.05) log(0.05)] ,'r-')
plot([log(0.05) log(0.05)] ,[log(2/factorial(8)) log(1)],'r-')
% set(gca,'XTick',[log(0.0005) log(0.005) log(0.05) log(0.5)])
% set(gca,'XTickLabel',['0.0005';' 0.005';'  0.05';'   0.5'])
% set(gca,'YTick',[log(0.0005) log(0.005) log(0.05) log(0.5)])
% set(gca,'YTickLabel',['0.0005';' 0.005';'  0.05';'   0.5'])

set(gca,'XTick',[log(0.0005) log(0.005) log(0.05) log(0.5)])
set(gca,'XTickLabel',[' ';' ';' ';' '])
% t = text([log(0.0005) log(0.005) log(0.05) log(0.5)],-10*ones(1,4)-0.5,['0.0005';' 0.005';' 0.05 ';'  0.5 ']);
% set(t,'HorizontalAlignment','center','VerticalAlignment','bottom','Rotation',0);
set(gca,'YTick',[log(0.0005) log(0.005) log(0.05) log(0.5)])
set(gca,'YTickLabel',[' ';' ';' ';' '])
% t = text(-10*ones(1,4),[log(0.0005) log(0.005) log(0.05) log(0.5)],['0.0005';' 0.005';' 0.05 ';'  0.5 ']);
% set(t,'HorizontalAlignment','center','VerticalAlignment','bottom','Rotation',90);
% 
% text( -11+3,-11,'P value linear track (log scale)')
% text(-11,-11+3, 'P value L-Shape track (log scale)','Rotation',90)


% for run=1:5
%     axes('position',[x_pos2(run) y_pos2(run) x_size2 y_size2]);
% end

