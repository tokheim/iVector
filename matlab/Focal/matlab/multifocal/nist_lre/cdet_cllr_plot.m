function cdet_cllr_plot(scores,classf,names,open_set);
% CDET_CLLR_PLOT: Bar-graph comparison, in terms of Cdet and Cllr of multiple recognizers.
%
%   Plots 6 quantities for every system:
%
%     act Cdet: actual average Cdet 
%               Scaled by 2, so that a useless, 50% error-rate system maps to 1.
%
%     opt Cdet: average Cdet, but where scores are re-calibrated by the evaluator.
%               Scaled by 2, so that a useless, 50% error-rate system maps to 1.
%               (It IS possible for this value to be worse than 'act Cdet', because
%                the calibration is optimized for multiclass Cllr, not specifically for Cdet.)
%
%     act Cllr: actual average 'detection' Cllr
%               Unscaled: A useless system maps to 1.  
%
%     opt Cllr: average Cllr, but where scores are re-calibrated by the evaluator.
%               Unscaled: A useless system maps to 1.
%               (It IS possible for this value to be worse than 'act Cllr', because
%                the calibration is optimized for multiclass Cllr, 
%                not specifically for detection Cllr.)
%
%     multi Cllr: actual multiclass Cllr 
%                 See MULTICLASS_CLLR.
%                 (Scaled by log_2 N, so that a useless system maps to 1.)  
%
%     min Cllr: multiclass Cllr, but where scores are re-calibrated by the evaluator.
%               See MULTICLASS_MIN_CLLR.
%               (Scaled by log_2 N, so that a useless system maps to 1.)  
%
%   Usage: 
%         CDET_CLLR_PLOT(scores, classf, names,open_set);
%         CDET_CLLR_PLOT({scores1, scores2, ...}, classf, {'sys1','sys2', ...});
%  
%   Input parameters:
%      scores  : 1 by K cell array, where each element is an N by T matrix. K >= 1
%                These are N scores for each of T trials for each of K input systems: 
%                  scores{k}(i,t) is the score of system k, for recognizing class i in trial t.
%                Scores should be calibrated to have relative log-likelihood nature: 
%                  scores{k,i,t} = log P(trial_t | class_i, system_k) - offset_tk
%
%      classf  : 1 by T row, with for closed-set, elements in {1,2, ..., N},
%                or for open-set, elements in {1,2, ..., N+1}.
%                These are the true class labels for the trials.
%
%      open_set: flag to select closed-set or open-set conditions:
%                 0: closed-set. 
%                 1: open-set. 
%                 (optional, default = 0)  
if nargin<4
   open_set = 0;
end;
quiet = 1;

K = length(scores);
if length(names) ~= K
   error('illegal parameter: lengths of ''scores'' and ''names'' should be equal');
end;

[N,T] = size(scores{1});




raw_cdet = zeros(1,K);
opt_cdet = zeros(1,K);

raw_cllr = zeros(1,K);
opt_cllr = zeros(1,K);
multi_cllr = zeros(1,K);
min_cllr = zeros(1,K);

norm = log(N)/log(2);

for k=1:K
   [min_cllr(k),multi_cllr(k),cal_loss,ref_loss,offset,scale] = multiclass_min_cllr(scores{k},classf);   
   opt_loglh = apply_nary_lin_fusion(scores{k},scale,offset);
   [raw_cdet(k),raw_cllr(k)] = avg_cdet_and_cllr(scores{k},classf,open_set);  
   [opt_cdet(k),opt_cllr(k)] = avg_cdet_and_cllr(opt_loglh,classf,open_set);  
   fprintf('sys %s: avgCdet=(%g,%g); avgCllr=(%g,%g); multiCllr=(%g,%g)\n',names{k},raw_cdet(k),opt_cdet(k),raw_cllr(k),opt_cllr(k),multi_cllr(k)/norm,min_cllr(k)/norm);
end;

colormap([ 1,0,0; 0,1,0; 1,0,1; 0,1,1; 0,0,1; 1,1,0 ]);
bar([2*raw_cdet;2*opt_cdet;raw_cllr;opt_cllr;multi_cllr/norm;min_cllr/norm]');
title('Cdet and Cllr');
xlabel('system');
ylabel('2*avg C_{det} , avg C_{llr}, mult C_{llr}');
set(gca,'xtick',[1:K]);
set(gca,'xticklabel',names);
grid;
legend('act Cdet','opt Cdet','act Cllr','opt Cllr','multi Cllr','min Cllr');



