function loglh = apply_hlda_backend(data,Trans,CC,Mu);
% APPLY_HLDA_BACKEND: applies (quadratic) HLDA gaussian backend to new data.
%
% Usage: loglh = APPLY_HLDA_BACKEND(data,Trans,CC,Mu);
%
% Input parameters: 
%
%      data         : D by T array of T data points of dimension D.
%
%      Trans,CC,Mu  : parameters of backend as 
%                     trained by TRAIN_HLDA_BACKEND().
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
 



loglh = apply_quadratic_backend(Trans*data,CC,Mu);

