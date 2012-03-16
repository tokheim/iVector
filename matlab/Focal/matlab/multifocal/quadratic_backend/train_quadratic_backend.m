function [CC,Mu] = train_quadratic_backend(data,classf,regularization,smoothing);
% TRAIN_QUADRATIC_BACKEND: Train parameters for quadratic (heteroscedastic Gaussian) backend,
%                          with optional PPCA or FA regularization of class covariances
%                          and with optional back-off smoothing between class-covariances. 
%   
%   Regularization:
%      PPCA = probabilistic principal component analysis
%      FA = factor analysis
%
%   Smoothing: averages (regularized) class covariances, with common within-class covariance,
%              using a specified smoothing constant. 
%
%   Input parameters:
%
%      data: a D by T array, where each of T data points is an D-dimensional column vector
%
%      classf  : 1 by T row-vector, with elements ranging from 1..N.
%                These are the true class labels for every data point.
%       
%      regularization: (optional                                                             ), 
%                      (default = no regularization i.e. unconstained ML covariance estimates)
%                      cell array, can be:
%                        {'ppca',rank}, where rank is the PPCA rank. See PPCA_COVARIANCE().
%                                       Every class covariance is estimated using PPCA of this rank.
%
%                        {'fa',rank}, where rank is the FA rank. See FA_COVARIANCE().
%                                       Every class covariance is estimated using FA of this rank.
%
%      smoothing: (optional, defalt = 0 = no smoothing)
%                 degree of smoothing: a scalar between 0 and 1
%                   0 implies no smoothing. 
%                   1 implies maximum smoothing, so that all class covariances become equal
%                     to common, within-class covariance and back-end becomes linear.
%
%    Output parameters:
%
%      CC     : 1 by N, cell array of Class Covariance matrices.
%
%      Mu    :  D by N matrix of class means 


if nargin<3
   regularization = {}; % no regularization
end;

if nargin<4
   smoothing = 0; % no smoothing
end;


[CC,counts,Mu] = class_covariances(data,classf,regularization,smoothing);
