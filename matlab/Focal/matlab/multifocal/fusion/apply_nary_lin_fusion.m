function loglh = apply_nary_lin_fusion(scores,alpha,beta);
%  NARY_FUSION: Calibrated linear fusion of scores of K input systems, to recognize N classes.
%
%  Usage:
%   loglh = NARY_FUSION(scores,alpha,beta);
%   loglh = NARY_FUSION(scores,alpha);
%
%  Input parameters:
%      scores  : 1 by K cell array, where each element is an N by T matrix.
%                These are N scores for each of T trials for each of K input systems. 
%
%      alpha  : K-component vector; combination weights for K input systems
%
%      beta   : N-component column vector; score offsets
%               optional, default = zeros(N,1)
%
%  Output parameters:
%      loglh: N by T matrix of relative log-likelihoods, 
%             (N log-likelihoods for each of T trials)
%             It is computed as:
%
%               loglh(i,t) = sum_k alpha(k) * scores{k}(i,t) + beta(i)
%             
%             If scores behave and alpha and beta are well-trained, then loglh has 
%             the following interpretation:
%
%               loglh(i,t) = log P(trail_t | class_i) - offset_t, 
%
%             where:
%               log denotes natural logarithm
%               offset_t is an unspecified real constant that may vary by trial
% 
%
%
%  See also:  TRAIN_NARY_LLR_FUSION: For supervised training of alpha and beta coefficients.
%             LOGLH2POSTERIOR: To use the log-likelihood output from this function
%                              to compute posteriors for the recognized classes.

if ~iscell(scores)
   scores = {scores};
end;

K = length(scores);
[N,T] = size(scores{1});

if length(alpha) ~= K,
   error('scores and alpha are incompatible');
end;

if nargin<3 | isempty(beta)
   beta = zeros(N,1);
elseif length(beta)~=N
   error('scores and beta are incompatible');
end;

loglh = repmat(beta,1,T);
for k=1:K;
   loglh = loglh + alpha(k)*scores{k};
end;
