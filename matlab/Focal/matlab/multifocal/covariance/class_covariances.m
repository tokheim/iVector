function [CC,counts,Mu,CWCC] = class_covariances(data,classf,regularization,smoothing);
% CLASS_COVARIANCES: Within-class covariance estimates, one for each labeled class,
%                    with optional PPCA or FA regularization and optional back-off smoothing. 
%
%   Regularization:
%      PPCA = probabilistic principal component analysis
%      FA = factor analysis
%
%   Smoothing: Each smoothed class covariance is convex combination of 
%              (regularized) class covariances, with common within-class covariance:
%              SmoothedCC = smoothing*CWCC + (1-smoothing)*CC
%
%    Usage:  
%                          CC = CLASS_COVARIANCES(data,classf);
%                 [CC,counts] = CLASS_COVARIANCES(data,classf,regularization);
%              [CC,counts,Mu] = CLASS_COVARIANCES(data,classf,regularization,smoothing);
%         [CC,counts,Mu,CWCC] = CLASS_COVARIANCES(data,classf,...);
%    
%    Input parameters: 
%
%      data: a D by T array, where each of T data points is an D-dimensional column vector
%
%      classf  : 1 by T row-vector, with elements ranging from 1..N.
%                These are the true class labels for every data point.
%       
%      regularization: cell array, can be:
%                        {'ppca',rank}, where rank is the PPCA rank. See PPCA_COVARIANCE().
%                                       Every class covariance is estimated using PPCA of this rank.
%
%                        {'fa',rank}, where rank is the FA rank. See FA_COVARIANCE().
%                                       Every class covariance is estimated using FA of this rank.
%                        {}, no regularization = unconstrained ML covariance estimation.
%                        (optional, parameter absent == {})
%
%      smoothing: (optional, defalt = 0 = no smoothing)
%                 degree of smoothing: a scalar between 0 and 1
%                   0 implies no smoothing. 
%                   1 implies maximum smoothing, so that all class covariances become equal
%                     to common, within-class covariance.
%
%    Output parameters:
%
%      CC     : 1 by N, cell array of Class Covariance matrices.
%
%      Mu    : (optional) D by N matrix of class means 
%
%      counts: (optional) N-vector of class counts 
%
%      CWCC  : (optional) Common, within-class, covariance, symmetric D by D, pos(semi)-def
%              This is the class-count-weighted-average of the CC covariances.
              


[D,T] = size(data);
if length(classf)~=T
   error('data and classf incompatible')
end;
if min(classf)~=1
   error('illegal parameter classf')
end;
N = max(classf);

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

if nargin<4
   smoothing = 0;
elseif (smoothing<0) | (smoothing>1)
   error('illegal parameter smoothing, must be scalar in interval [0,1]')
end;




CC = cell(1,N);
Mu = zeros(D,N); %class means
S = zeros(D); %total scatter matrix
counts = zeros(1,N);
for i=1:N
   f=find(classf==i);
   counts(i) = length(f);
   if counts(i) == 0
      fprintf('%i trials of class %i\n',counts(i),i);
      error('zero trial count')
   end;
   data_i = data(:,f);
   if do_ppca
      [Cml,Mu(:,i)] = ml_covariance(data_i); 
      CC{i} = ppca(Cml,rank);
   elseif do_fa
      [CC{i},Mu(:,i)] = fa_covariance(data_i,rank);   
   else % no regularization, just do ML
      [CC{i},Mu(:,i)] = ml_covariance(data_i); 
    end; 
   S = S + CC{i}*counts(i);
end;

% common within-class covariance estimate
CWCC = S/T;

if smoothing>0
   for i=1:N;
      CC{i} = smoothing*CWCC+(1-smoothing)*CC{i};
   end;
end;


