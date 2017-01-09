%clear all
%close all
%load('D:\Alon\Work\Butterflies\Ver0\fig3\HeadDirection\RingEstimation\output_Mouse28-140313\output\all\wake\full_reduced_data.mat')
start_value=0.004;
end_value=0.023;
slope_one_color=[1 0.2 0.8];

DataForDimEst=full_reduced_data{3}(1:10:end,2:10);
NumOfDataPoints=size(DataForDimEst,1);
RepAndPermData=permute(repmat(DataForDimEst,[1 1 NumOfDataPoints]),[1 3 2]);
diff_RepAndPermData=RepAndPermData-permute(RepAndPermData,[2 1 3]);
diff_RepAndPermData=(sum(diff_RepAndPermData.^2,3)).^0.5;
diff_RepAndPermData(~~eye(NumOfDataPoints))=NaN;
figure (401)
hold on
hist(diff_RepAndPermData(:),1000)

[a b]=hist(diff_RepAndPermData(:),1000);
[~,start_bin_ind]=min(abs(b-start_value))
[~,end_bin_ind]=min(abs(b-end_value));
plot(start_value*[1 1],[0 1.1*max(a)],'r-')
plot(end_value*[1 1],[0 1.1*max(a)],'r-')
ylim([0 1.1*max(a)])

%figure, plot(b,cumsum(a),'*')

figure(402)
plot(log(b),log(cumsum(a)),'r*')
hold on
% plot([log(b(end))-10 log(b(end))],[log(sum(a))-10 log(sum(a))],'r-')
% plot([log(b(end))-10 log(b(end))],[log(sum(a))-20 log(sum(a))],'r-')

vy=cumsum(a(start_bin_ind:end));
vx=b(start_bin_ind:end)-b(start_bin_ind-1);
relevant_ind=1:(end_bin_ind-start_bin_ind+1);
figure (403)
hold on
plot(log(vx(relevant_ind)),log(vy(relevant_ind)),'r.')
plot(log(vx(relevant_ind(end)+1:end)),log(vy(relevant_ind(end)+1:end)),'r-')

% plot([log(vx(end))-10 log(vx(end))],[log(vy(end))-10 log(vy(end))],'r-')
% plot([log(vx(end))-10 log(vx(end))],[log(vy(end))-20 log(vy(end))],'r-')

% DataForDimEst=v2_data_type1(:,2:10);
% NumOfDataPoints=size(DataForDimEst,1);
% RepAndPermData=permute(repmat(DataForDimEst,[1 1 NumOfDataPoints]),[1 3 2]);
% diff_RepAndPermData=RepAndPermData-permute(RepAndPermData,[2 1 3]);
% diff_RepAndPermData=(sum(diff_RepAndPermData.^2,3)).^0.5;
% diff_RepAndPermData(~~eye(NumOfDataPoints))=NaN;
% figure(504)
% hist(diff_RepAndPermData(:),1000)
% [a b]=hist(diff_RepAndPermData(:),1000);
% %figure, plot(b,cumsum(a),'*')
% 
% figure(501)
% plot(log(b),log(cumsum(a)),'b*')
% hold on
% % plot([log(b(end))-10 log(b(end))],[log(sum(a))-10 log(sum(a))],'r-')
% % plot([log(b(end))-10 log(b(end))],[log(sum(a))-20 log(sum(a))],'r-')
% 
% vy=cumsum(a(2:end));
% vx=b(2:end);
% figure (502)
% plot(log(vx),log(vy),'b*')
% hold on
% % plot([log(vx(end))-10 log(vx(end))],[log(vy(end))-10 log(vy(end))],'r-')
% % plot([log(vx(end))-10 log(vx(end))],[log(vy(end))-20 log(vy(end))],'r-')
% 
figure (402)
% for run1=6:27
%     plot([-11 -1],run1+1*[-11 -1],'-b')
% end
for run2=6:30
    plot([-11 -1],run2+1*[-11 -1],'-','color',slope_one_color)
end
xlim([-11 -2])
ylim([4 20])

figure (403)
for run1=6:30
    plot([-11 -1],run1+1*[-11 -1],'-','color',slope_one_color)
end
% for run2=8:38
%     plot([-11 -1],run2+2*[-11 -1],'-r')
% end
xlim([-11 -2])
ylim([4 20])

