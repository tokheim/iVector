function [C,mu,sigma,W] = ppca_covariance(data,r);
% PPCA_COVARIANCE: PPCA-constrained, maximum-likelihood covariance estimate.
%
%
%    PPCA Modeling assumption: Covariance  = sigma I + W*W', where 
%                              I is D-by-D identity and
%                              W is D-by-r, with r<d.  
%
%
%    Usage:  C = PPCA_COVARIANCE(data,r);
%            [C,sigma,W] = PPCA_COVARIANCE(Cun,r);
%    
%    Input parameters: 
%
%      data: a D by T array, where each data point is a D-dimensional column vector
%
%      r   : an integer, where 1< r < D. 
%            Special cases: r = 0 gives isotropic covariance C = sigma I
%                           r = D gives unconstrained covariance C = Cun.
%            The smaller r is made, the stronger the constraint and therefore the 
%            'regularization' effect. If the data is scarce (T small), then choosing a small 
%            r is likely to be beneficial.
%
%
%    Output parameters:
%
%      C: PPCA-regularized covariance matrix.
%         Symmetric, D by D, positive definite (or semi-definite) matrix.
%         (If there are zero (or very small) eigenvalues, try to decrease r.)
% 
%      mu: (optional) mean of data
%
%      sigma,W: (optional), model parameters, where C = sigma I + W*W'.


[Cml,mu] = ml_covariance(data);

[C,sigma,W] = ppca(Cml,r);

