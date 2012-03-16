function loglh = apply_linear_backend(data,Trans,offset);
% APPLY_LINEAR_BACKEND: applies linear (homoscedastic gaussian) backend to new data.
%
% Usage: loglh = APPLY_LINEAR_BACKEND(data,Trans,offset);
%
% Input parameters: 
%
%      data         : D by T array of T data points of dimension D.
%
%      Trans,offset : parameters of backend as 
%                     trained by TRAIN_LINEAR_BACKEND().
%
% Output parameters:
%
%      loglh: N by T matrix of relative log-likelihoods, 
%             (log-likelihoods for each of N classes, given each of T trials)
%
%             loglh has the following interpretation:
%
%               loglh(i,t) = log P(trail_t | class_i) - offset_t, 
%
%             where:
%               log denotes natural logarithm
%               offset_t is an unspecified real constant that may vary by trial
 




[D,T] = size(data);
loglh = Trans*data+repmat(offset,1,T);