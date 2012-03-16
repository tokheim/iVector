function [refinement_loss,total_loss,calibration_loss,reference_loss,offset,scale] = multiclass_min_cllr(loglh,classf,quiet);
%
% MULTICLASS_MIN_CLLR: Measure of refinement of log-likelihood-ratio-representation of N-class 
%                recognition information.
%
% 		Usage:
%       [list of output parameters] = MULTICLASS_MIN_CLLR(loglh, classf, quiet);
%
%     Input parameters:
%
%       loglh  : N by T matrix of relative log-likelihoods, 
%                There are N log-likelihoods for each of T trials:
%                  loglh(i,t) = log P(trail_t | class_i) - offset_t, 
%                where:
%                   log denotes natural logarithm
%                   offset_t is an unspecified real constant that may vary by trial
%
%       classf : a row of T integers in the range 1,2,...,N, indicating the true class 
%                of each trial.
%
%       quiet  : 0 show optimization iterations
%                1 don't show optimization
%                optional: default  1
%
%     Output parameters:
%
%         refinement_loss:  
%                     average cost in bits per trial, but minimized, w.r.t. scale and 
%                     offset_vector, so that 
%                             opt_loglh = scale * loglh + offset_vector
%                     Range: 0 .. approx. log(N)/log(2)
%                     Note: minimization is done with access to the truth reference and
%                           is performed by calling TRAIN_NARY_LLR_FUSION(loglh,classf,...).
%
%         total_loss: average cost in bits per trial of using loglh to recognize the classes
%                     present in the evaluation data. This is just:
%                       total_loss = multiclass_cllr(loglh,classf). 
%                     Range: 0 .. inf.
%                     (OPTIONAL)
%
%         calibration_loss: total_loss - refinement_loss  
%                           Range: 0 .. inf
%                           (OPTIONAL)
%         
%         reference_loss: log(N)/log(2) 
%                         is the reference value for a default system 
%                         that makes recognition decisions based just 
%                         on the prior, i.e. when loglh = zeros(N,T). 
%                         (OPTIONAL)
%
%         scale: the scalar multiplier that was found to optimize log-likelihoods
%                (OPTIONAL)
%
%         offset: the N-vector offset that was found to optimize log-likelihoods
%                 (OPTIONAL)
%         
%         shift_loss: component of calibration_loss due to shift: 
%                     calibration_loss = scaling_loss + shift_loss         ,
%                     (OPTIONAL) 
%
%         scaling_loss: component of calibration_loss due to scaling:         ,
%                       calibration_loss = scaling_loss + shift_loss         ,
%                      (OPTIONAL) 
%
%
%      See also:  MULTICLASS_CLLR, TRAIN_NARY_LLR_FUSION


if nargin<3
   quiet = 1;
end;


[N,T]=size(loglh);
if (min(classf)~=1) | (max(classf)~=N) | (length(classf)~=T), 
   error('classf and loglh incompatible'); 
end;

% alpha = 1 should be close to optimum if input is well-calibrated, but when calibration is 
% off, this initialization seems to make CG-optimizer suffer. So we use alpha = 0.
[scale,offset] = train_nary_llr_fusion({loglh},classf,0,1.0e-6,ones(N,1)/N,0,zeros(N,1),quiet);

opt_loglh = apply_nary_lin_fusion({loglh},scale,offset);

total_loss = multiclass_cllr(loglh,classf);
refinement_loss = multiclass_cllr(opt_loglh,classf);

if (refinement_loss > total_loss) % try to fix
   if ~quiet
      fprintf('logistic regression optimization failed, trying again\n');
   end;
   % now try alpha = 1
   [scale,offset] = train_nary_llr_fusion({loglh},classf,0,1.0e-6,ones(N,1)/N,1,zeros(N,1),quiet);
   opt_loglh = apply_nary_lin_fusion({loglh},scale,offset);
   refinement_loss = multiclass_cllr(opt_loglh,classf);
end;


if (refinement_loss > total_loss) % give up gracefully
   warning('logistic regression optimization failed, setting refinement_loss = total_loss');
   scale = 1;
   offset = zeros(N,1);
   refinement_loss = total_loss;
end;

calibration_loss = total_loss-refinement_loss;
reference_loss = log(N)/log(2);



