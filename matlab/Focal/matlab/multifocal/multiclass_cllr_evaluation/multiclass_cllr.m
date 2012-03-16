function cost = multiclass_cllr(loglh,classf);
% MULTICLASS_CLLR: Measure of goodness of log-likelihood-ratio-representation of N-class recognition
%        information.
%
% 		Usage:
%       cost = MULTICLASS_CLLR(loglh, classf);
%
%     Input parameters:
%       loglh  : N by T matrix of relative log-likelihoods, 
%                There are N log-likelihoods for each of T trials:
%                  loglh(i,t) = log P(trail_t | class_i) - offset_t, 
%                where:
%                   log denotes natural logarithm
%                   offset_t is an unspecified real constant that may vary by trial
%
%       classf : a row of T integers in the range 1,2,...,N, indicating the true class 
%                of each trial.
%
%     Output:
%         cost: average cost in bits per trial of using loglh to recognize the classes
%               present in the evaluation data.
%         
%           range: 
%             0 <= cost <= inf, where 
%               0 is perfect recognition and 
%               inf denotes very badly calibrated log-likelihoods
%           reference value:
%             cost = log(N)/log(2) is the reference value for a default system that makes 
%                                  recognition decisions based just on the prior, 
%                                  i.e. when loglh = zeros(N,T). 
%                                  Any cost > log(N)/log(2) indicates a useless 
%                                  recognizer with bad calibration. 
%
%      See also:  MULTICLASS_MIN_CLLR


[N,T]=size(loglh);
if (min(classf)~=1) | (max(classf)~=N) | (length(classf)~=T), 
   error('classf and loglh incompatible'); 
end;
cost=0;
lsm = logsoftmax(loglh);
for i=1:N;
   f = find(classf==i);
   cost = cost + mean( -lsm(i,f) );
end;
cost = cost / (N*log(2));
