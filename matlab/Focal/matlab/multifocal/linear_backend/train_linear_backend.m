function [Trans,offset] = train_linear_backend(data,classf,regularization);
% TRAIN_LINEAR_BACKEND: Train a linear (homoscedastic Gaussian) back-end, 
%                       with optional PPCA or FA regularization common-within-class covariance
%
%   Regularization:
%      PPCA = probabilistic principal component analysis
%      FA = factor analysis
%
%    Usage: [Trans,offset] = TRAIN_LINEAR_BACKEND(data,classf,regularization);
%
%    Input parameters: 
%
%      data    : a D by T array, where each of T data points is a D-dimensional column vector.
%
%      classf  : 1 by T row-vector, with elements ranging from 1..N, where there are N classes.
%                These are the true class labels for every data point.
%
%      regularization: (optional                                                             ), 
%                      (default = no regularization i.e. unconstained ML covariance estimate)
%                      cell array, can be:
%                        {'ppca',rank}, where rank is the PPCA rank. See PPCA_COVARIANCE().
%
%                        {'fa',rank}, where rank is the FA rank. See FA_COVARIANCE().
%
%    Output parameters: 
%      
%      Trans: An N-by-D matrix to project D-dimensional inputs to N-dimensional outputs.
%      offset: an N-dimensional offset vector, to be applied after projection.
%              To apply this backend to an input vector x, do: 
%                y = Trans*x+offset
      


do_ppca = 0;
do_fa = 0;
if (nargin >= 3) & ~isempty(regularization)
   if strcmpi( regularization{1}, 'ppca' )
      do_ppca = 1;
      rank = regularization{2};
   elseif strcmpi( regularization{1}, 'fa' )
      do_fa = 1;
      rank = regularization{2};
   else
      error('illegal regularization parameter');
   end;
end;


D = size(data,1);

if do_ppca   
   [C,counts,Mu,CWCC] = class_covariances(data,classf);
   CWCC = ppca(CWCC,rank); 
elseif do_fa
   [CWCC,Mu] = fa_cwc_covariance(data,classf,rank);
else % just do ML
   [C,counts,Mu,CWCC] = class_covariances(data,classf);
end;


[Trans,offset] = compose_linear_backend(CWCC,Mu);
