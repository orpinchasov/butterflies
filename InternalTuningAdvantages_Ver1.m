figure ;
%Some WYSIWYG options:
set(gcf,'DefaultAxesFontSize',10);
set(gcf,'DefaultAxesFontName','arial');
fig_size_y=10;
fig_size_x=20;
set(gcf,'PaperUnits','centimeters','PaperPosition',[5 2 fig_size_x fig_size_y]);
set(gcf,'PaperOrientation','portrait');
set(gcf,'Units','centimeters','Position',get(gcf,'paperPosition')+[15 5 0 0]); %+[4 2 0 0]);
LineWithValue=2;

start_x1=0.06;
start_x2=0.39;
start_x3=0.73;

% start_y1a=0.075;
% start_y1b=0.275;
% start_y1c=0.575;
% start_y1d=0.775;
start_y1a=0.1;
start_y1b=0.6;
start_y2a=0.1;
start_y2b=0.425;
start_y3a=0.1;
start_y3b=0.575;

size_x1=0.24;
size_x2=0.25;
size_x3=0.18;

size_y1=0.3;
size_y2a=0.2;
size_y2b=0.5;
size_y3=0.35;

% Part A, dealing with errors of thge animal
ColorMatPartA=[1 0 0;0 1 0;0 0 1;1 0 1;0 0 0];
mean_vec_PartA=[0.2 0.6];
var_vec_PartA=[0.09 0.06];
hight_vec_PartA=[1 1];
x_values_PartA=[0:0.001:1];
p_partA=0.65;
y_values_2_1_PartB=hight_vec_PartA(1)*normpdf(x_values_PartA,mean_vec_PartA(1),var_vec_PartA(1));
y_values_2_2_PartB=hight_vec_PartA(2)*normpdf(x_values_PartA,mean_vec_PartA(2),var_vec_PartA(2));
y_values_1_1_PartB=p_partA*y_values_2_1_PartB+(1-p_partA)*y_values_2_1_PartB(end:-1:1);
y_values_1_2_PartB=p_partA*y_values_2_2_PartB+(1-p_partA)*y_values_2_2_PartB(end:-1:1);
axes('position',[start_x1 start_y1a size_x1 size_y1]);
plot(x_values_PartA,y_values_1_1_PartB,'-','color',ColorMatPartA(1,:),'LineWidth',LineWithValue)
hold on
plot(x_values_PartA,y_values_1_2_PartB,'-','color',ColorMatPartA(2,:),'LineWidth',LineWithValue)
xlim([0 1])
ylim([0 8])
xlabel('Position')
ylabel('Neural activity')

axes('position',[start_x1 start_y1b size_x1 size_y1]);
plot(x_values_PartA,y_values_2_1_PartB,'-','color',ColorMatPartA(1,:),'LineWidth',LineWithValue)
hold on
plot(x_values_PartA,y_values_2_2_PartB,'-','color',ColorMatPartA(2,:),'LineWidth',LineWithValue)
xlim([0 1])
ylim([0 8])
xlabel('Position')
ylabel('Neural activity')

% Part B, tuning to mutliple variables
ColorMatPartB=[1 0 0;0 1 0;0 0 1;1 0 1;0 0 0];
x_vec_PartB=[0.17 0.4 0.5 0.6 0.75];
y_vec_PartB=[0.8 0.2 0.5 0.3 0.7];
var_vec_PartB=[0.15 0.15 0.2 0.25 0.27];
x_values_PartB=[0:0.01:1];
axes('position',[start_x2 start_y2a size_x2 size_y2a]);
for run_PartB=1:length(x_vec_PartB)
    plot(x_values_PartB,normpdf(x_values_PartB,x_vec_PartB(run_PartB),0.5*var_vec_PartB(run_PartB)),'color',ColorMatPartB(run_PartB,:),'LineWidth',LineWithValue)
    hold on
end
xlim([0 1])
ylim([0 7])
box off
xlabel('Variable 1')
ylabel('Neural activity')

axes('position',[start_x2 start_y2b size_x2 size_y2b]);
for run_PartB=1:length(x_vec_PartB)
    viscircles([x_vec_PartB(run_PartB) y_vec_PartB(run_PartB)],var_vec_PartB(run_PartB),'EdgeColor',ColorMatPartB(run_PartB,:));
    hold on
end
xlim([0 1])
ylim([0 1])
box on
xlabel('Variable 1')
ylabel('Variable 2')


% Part C, neural synonyms
N=20;
randn('seed',0)
% random_dots1_x=0.75;
% random_dots1_y=0.25;
% random_dots2_x=0.2;
% random_dots2_y=0.7;
% random_dots_std=0.05;
% random_dots1=[random_dots1_x*ones(N,1) random_dots1_y*ones(N,1)]+random_dots_std*randn(N,2);
% random_dots2=[random_dots2_x*ones(N,1) random_dots2_y*ones(N,1)]+random_dots_std*randn(N,2);
% axes('position',[start_x3 start_y3 size_x3 size_y3]);
% plot(random_dots1(:,1),random_dots1(:,2),'k.')
% hold on
% plot(random_dots2(:,1),random_dots2(:,2),'k.')
% plot(0.5*(random_dots1_x+random_dots2_x),0.5*(random_dots1_y+random_dots2_y),'r+','MarkerSize',10,'LineWidth',LineWithValue)
% xlim([0 1])
% ylim([0 1])
% box off
% xlabel('Neuron 1 activity given condition')
% ylabel('Neuron 2 activity given condition')

random_dots1a_x=0.15;
random_dots1a_y=0.15;
random_dots1b_x=0.6;
random_dots1b_y=0.85;
random_dots2a_x=0.15;
random_dots2a_y=0.70;
random_dots2b_x=0.75;
random_dots2b_y=0.20;
random_dots3a_x=0.45;
random_dots3a_y=0.5;
random_dots3b_x=0.9;
random_dots3b_y=0.9;
random_dots_std=0.05;
%creating the distribtions of the data points
random_dots1a=[random_dots1a_x*ones(N,1) random_dots1a_y*ones(N,1)]+random_dots_std*randn(N,2);
random_dots1b=[random_dots1b_x*ones(N,1) random_dots1b_y*ones(N,1)]+random_dots_std*randn(N,2);
random_dots2a=[random_dots2a_x*ones(N,1) random_dots2a_y*ones(N,1)]+random_dots_std*randn(N,2);
random_dots2b=[random_dots2b_x*ones(N,1) random_dots2b_y*ones(N,1)]+random_dots_std*randn(N,2);
random_dots3a=[random_dots3a_x*ones(N,1) random_dots3a_y*ones(N,1)]+random_dots_std*randn(N,2);
random_dots3b=[random_dots3b_x*ones(N,1) random_dots3b_y*ones(N,1)]+random_dots_std*randn(N,2);
%claculating the naive average of each type
center_1=0.5*[random_dots1a_x+random_dots1b_x random_dots1a_y+random_dots1b_y];
center_2=0.5*[random_dots2a_x+random_dots2b_x random_dots2a_y+random_dots2b_y];
center_3=0.5*[random_dots3a_x+random_dots3b_x random_dots3a_y+random_dots3b_y];

axes('position',[start_x3 start_y3b size_x3 size_y3]);
hold on
for run_line=1:5
    plot([0 1],run_line*[0.2 0.2],'k--')
    plot(run_line*[0.2 0.2],[0 1],'k--')  
end
plot(random_dots1a(:,1),random_dots1a(:,2),'b.')
plot(random_dots1b(:,1),random_dots1b(:,2),'b.')
plot(random_dots2a(:,1),random_dots2a(:,2),'r.')
plot(random_dots2b(:,1),random_dots2b(:,2),'r.')
plot(random_dots3a(:,1),random_dots3a(:,2),'g.')
plot(random_dots3b(:,1),random_dots3b(:,2),'g.')
xlim([0 1])
ylim([0 1])
xlabel('Neuron 1 activity')
ylabel('Neuron 2 activity')

axes('position',[start_x3 start_y3a size_x3 size_y3]);
hold on
for run_line=1:5
    plot([0 1],run_line*[0.2 0.2],'k--')
    plot(run_line*[0.2 0.2],[0 1],'k--')  
end
plot(center_1(1),center_1(2),'b+','MarkerSize',10,'LineWidth',LineWithValue)
plot(center_2(1),center_2(2),'r+','MarkerSize',10,'LineWidth',LineWithValue)
plot(center_3(1),center_3(2),'g+','MarkerSize',10,'LineWidth',LineWithValue)
xlim([0 1])
ylim([0 1])
xlabel('Neuron 1 activity')
ylabel('Neuron 2 activity')