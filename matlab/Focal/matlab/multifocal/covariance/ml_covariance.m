function [C,mu] = ml_covariance(data);
% ML_COVARIANCE: Unconstrained, maximum-likelihood covariance estimate,
%                assuming a mulitvariate Gaussian data model.
%
%    Usage:  C = ML_COVARIANCE(data);
%    
%    Input parameters: 
%
%      data: an D by T array, where each data point is an D-dimensional column vector
%
%    Output parameters:
%
%      C: symmetric, D by D, positive definite (or semi-definite) matrix   
%      mu: (optional) D-dimensional mean of data


[D,T] = size(data);

mu = mean(data,2);
%C = data*data'/T-mu*mu';
data = data-repmat(mu,1,T);
C = data*data'/T;
