function [HeadDirectionDiff HeadDirectionDiffFreqOfShuffle]=...
    ShuffleInternalVsExternalHeadDirection_Ver1(External_angle,Internal_angle,NumOfShuffles,NumOfBins)
% maybe to give the binning as input instead to set it as [-pi*(1-1/64):pi/32:pi*(1-1/64)]

BinCentersVec=[-pi*(1-1/NumOfBins):2*pi/NumOfBins:pi*(1-1/NumOfBins)];

NumOfShuffles=NumOfShuffles+1;
RotationFactorIntVsExt=[-pi:(0.1*pi):pi];
Shuffled_HeadDirection_Mean_ErrorVec=nan(1,NumOfShuffles);
Current_Shuffled_HeadDirection_Error_mat=nan(2*length(RotationFactorIntVsExt),length(Internal_angle));
All_Errors_Shuffled_HeadDirection=nan(NumOfShuffles,length(Internal_angle));
for run_shuffle=1:NumOfShuffles
    if run_shuffle==1
        Shuffled_Internal_angle=Internal_angle;
    else
        Shuffled_Internal_angle=Internal_angle(randperm(length(Internal_angle)));
    end
    
    %Internal_angle
    for ReflectionFactorBeteenMice=[-1 1]
        for run_RotationFactor=1:length(RotationFactorIntVsExt)
            RotationFactorBeteenMice=RotationFactorIntVsExt(run_RotationFactor);
            Shuffled_corrected_Internal_angle=...
                ReflectionFactorBeteenMice*Shuffled_Internal_angle+RotationFactorBeteenMice;
            Shuffled_HeadDirection_Error=mod(External_angle-Shuffled_corrected_Internal_angle+pi,2*pi)-pi;
            current_vec_ind=run_RotationFactor+0.5*(1+ReflectionFactorBeteenMice)*length(RotationFactorIntVsExt);
            Current_Shuffled_HeadDirection_Error_mat(current_vec_ind,:)=Shuffled_HeadDirection_Error;
            
        end
    end
    %sum(isnan(Current_Shuffled_HeadDirection_Error_mat(:)))
    Current_Shuffled_HeadDirection_Error_score=nanmean(Current_Shuffled_HeadDirection_Error_mat.^2,2);
    [Shuffled_HeadDirection_Mean_Error,ind_of_least_error_shuffle]=min(Current_Shuffled_HeadDirection_Error_score);
    Shuffled_HeadDirection_ErrorVec(run_shuffle)=Shuffled_HeadDirection_Mean_Error;
    All_Errors_Shuffled_HeadDirection(run_shuffle,:)=Current_Shuffled_HeadDirection_Error_mat(ind_of_least_error_shuffle,:);      
end
All_Errors_Shuffled_HeadDirectionVec=All_Errors_Shuffled_HeadDirection(2:end,:);
All_Errors_Shuffled_HeadDirectionVec=All_Errors_Shuffled_HeadDirectionVec(:);
[hist_AcrossMiceShuffled_HD_a hist_AcrossMiceShuffled_HD_b]=hist(All_Errors_Shuffled_HeadDirectionVec,BinCentersVec);

HeadDirectionDiff=hist_AcrossMiceShuffled_HD_b;
HeadDirectionDiffFreqOfShuffle=hist_AcrossMiceShuffled_HD_a./sum(~isnan(All_Errors_Shuffled_HeadDirection(:)));

%%
% figure;
% hold on
% [hist_AcrossMice_HD_a hist_AcrossMice_HD_b]=hist(HeadDirection_Error,[-pi*(1-1/64):pi/32:pi*(1-1/64)]);
% plot(hist_AcrossMice_HD_b, hist_AcrossMice_HD_a./sum(~isnan(HeadDirection_Error)), 'b-')
% All_Errors_Shuffled_HeadDirectionVec=All_Errors_Shuffled_HeadDirection(2:end,:);
% All_Errors_Shuffled_HeadDirectionVec=All_Errors_Shuffled_HeadDirectionVec(:);
% [hist_AcrossMiceShuffled_HD_a hist_AcrossMiceShuffled_HD_b]=hist(All_Errors_Shuffled_HeadDirectionVec,[-pi*(1-1/64):pi/32:pi*(1-1/64)]);
% plot(hist_AcrossMiceShuffled_HD_b, hist_AcrossMiceShuffled_HD_a./sum(~isnan(All_Errors_Shuffled_HeadDirection(:))), 'r-')
% xlim([-pi pi])
