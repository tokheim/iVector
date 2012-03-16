function [C,mu] = fa_covariance(data,r,num_iter);
% FA_COVARIANCE: Constrained, maximum-likelihood, factor-analysis covariance estimate,
%                assuming a mulitvariate Gaussian data model.
%
%
%    FA Modeling assumption: Covariance  = Psi + W*W', where 
%                              Psi is D-by-D diagonal and
%                              W is D-by-r, with 0 <= r =< d.  
%
%
%    Usage:  [C,mu] = FA_COVARIANCE(data,r);
%                 C = FA_COVARIANCE(data,r,num_iter);
%    
%    Input parameters: 
%
%      data: a D by T array, where each data point is an D-dimensional column vector
%
%      r   : an integer, where 0 =< r <= D. 
%            The smaller r is made, the stronger the constraint and therefore the 
%            'regularization' effect. If the data is scarce (T small), then choosing a small 
%            r is likely to be beneficial.
%            Special cases:
%              r = 0: C = Psi. This gives a pure diagonal covariance estimate.
%              r = D-1, or r = D: C = Cml. This gives the unconstrained ML covariance estimate. 
%
%      num_iter: The maximum number of EM-algorithm iterations to run to maximize the 
%                likelihood of the factor analysis model. 
%                (optional: see FACTOR_ANALYSIS_EM for default)
%                If num_iter is specified, iteration progress is printed, otherwise quiet.
%
%
%    Output parameters:
%
%      C: symmetric, D by D, positive definite (or semi-definite) matrix.
%         (If there are zero (or very small) eigenvalues, try to decrease r.)
%
%      mu: (optional) mean of data

[D,T] = size(data);

mu = mean(data,2);

if r==0 % just return diagonal ml covariance estimate
   V = sum(data.^2,2)/T-mu.^2;
   C = diag(V);
   return;
end;


if r==D % just return full ml covariance
   C = ml_covariance(data);
   return;
end;

if (r<1) | (r>D)
   error('illegal parameter r');
end;


if nargin<3
   [W,Psi] = factor_analysis_em(data,r);
else
   [W,Psi] = factor_analysis_em(data,r,num_iter);
end;   

C = diag(Psi) + W*W'; 