function [CWCC,Mu] = fa_cwc_covariance(data,classf,r,num_iter);
% FA_CWC_COVARIANCE: Factor-analysis-constrained, maximum-likelihood, common within-class covariance estimate.
%
%    Usage:     
%               CWCC      = FA_CWC_COVARIANCE(data,classf,r,num_iter);
%               [CWCC,Mu] = FA_CWC_COVARIANCE(data,classf,r);
%    
%    Input parameters: 
%
%      data    : a D by T array, where each of T data points is an D-dimensional column vector.
%
%      classf  : 1 by T row-vector, with elements ranging from 1..N.
%                These are the true class labels for every data point.
%
%      r       : an integer, where 1< r < D. 
%                The smaller r is made, the stronger the constraint and therefore the 
%                'regularization' effect. If the data is scarce (T small), then choosing a small 
%                r is likely to be beneficial.
%                Special cases:
%                  r = 0: C = Psi. This gives a pure diagonal covariance estimate.
%                  r = D-1, or r = D: C = Cml. This gives the unconstrained ML covariance estimate. 
%
%      num_iter: The maximum number of EM-algorithm iterations to run to maximize the 
%                likelihood of the factor analysis model. 
%                (optional: see FACTOR_ANALYSIS_EM for default)
%                If num_iter is specified, iteration progress is printed, otherwise quiet.
%
%    Output parameters:
%
%      CWCC: symmetric, D by D, positive definite (or semi-definite) matrix.
%            Common, within-class, covariance. 
%            (If there are zero (or very small) eigenvalues, try to decrease r.)
%
%      Mu: (optional) D by N matrix of class means 

[D,T] = size(data);
if length(classf)~=T
   error('data and classf incompatible')
end;
if min(classf)~=1
   error('illegal parameter classf')
end;
N = max(classf);

cdata = zeros(D,T); % centered data

Mu = zeros(D,N); %class means
for i=1:N
   f=find(classf==i);
   c = length(f);
   if c==0
      fprintf('%i trials of class %i\n',c,i);
      error('zero trial count')
   end;
   Mu(:,i) = mean(data(:,f),2);
   cdata(:,f) = data(:,f)-repmat(Mu(:,i),1,c);   
end;

% common within-class covariance estimate
if nargin<4
   CWCC = fa_covariance(cdata,r);
else
   CWCC = fa_covariance(cdata,r,num_iter);
end;




