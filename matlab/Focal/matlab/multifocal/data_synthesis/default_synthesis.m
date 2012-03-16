function [loglh,classf]=default_synthesis(no_classes,trialsPerCLass,inter_class_separation);
% DEFUALT_SYNTHESIS creates a matrix of multi-class log-likelihoods for a set of 
%                   synthetic data points. 
%
% [ll,classf] = DEFUALT_SYNTHESIS(no_classes,trialsPerCLassT,inter_class_separation);
% 
% INPUT parameters:
%   no_classes       : the number of classes to generate likelihoods for
%   trialsPerCLassT  : the number of random data points to generate in feature space for each class.
%   inter_class_separation: a positive number to control the separation of classes in feature space.
%                           Increase this number to make classes more recognizable. 
%                           (Try 2.0 to start.)
%
% OUTPUT parameters:
%   loglh  : a no_classes-BY-(no_classes*trialsPerCLassT) matrix of log-likelihoods
%   classf : a row of (no_classes*trialsPerCLassT) integers in the range 1..no_classes, 
%            denoting the class of each observation. 


% feature dimension = no of classes
% Indentity covariance
C = eye(no_classes); 

% mean of class i is separated by 'inter_class_separation' from origin
Mu = eye(no_classes) * inter_class_separation; 

[loglh,classf] = gauss_loglh(C,Mu,trialsPerCLass); 
