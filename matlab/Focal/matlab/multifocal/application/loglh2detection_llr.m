function llr = loglh2detection_llr(loglh,prior,target);
%  LOGLH2DETECTION_LLR: log-likelihood to detection log-likelihood-ratio.
%  
%
%  Usage:
%
%    llr = LOGLH2DETECTION_LLR(loglh,prior,target)
%
%  Input parameters:
%       loglh  : N by T matrix of relative log-likelihoods, 
%                There are N log-likelihoods for each of T trials:
%                  loglh(i,t) = log P(trail_t | class_i) - offset_t, 
%                where:
%                   log denotes natural logarithm
%                   offset_t is an unspecified real constant that may vary by trial
%       prior  : N-component prior probability distribution (positive, components sum to one).
%                The component prior(target) is ignored. The relative magnitudes of the
%                rest of the probabilities (for the non-target classes) are used.
%
%       target: an integer in the range 1..N, denoting the target class. The other classes are
%               assumed to be non-target classes.
%
%  Output:
%       llr = detection log-likelihood-ratios for every trial
%             (1 by T row vector )
%             llr(t) = log ( P(trial_t | target) / P(trial_t | non-target) )
%             


[N,T] = size(loglh);
prior = prior(:);
if length(prior) ~= N
   error('loglh and prior incompatible');
end;
if min(prior) <=0
   error('illegal prior: has negative or zero component(s)');
end;
if abs(log(sum(prior))) > 0.001
   error('illegal prior: does not sum to one');
end;

if (target<1) | (target>N)
   error('illegal parameter target');
end;

prior(target) = 0;
s = sum(prior);
prior = prior*0.5/s;
prior(target) = 0.5;



loglh = loglh+repmat(log(prior),1,T);
w = [1:target-1,target+1:N];
llr = loglh(target,:) - logsumexp(loglh(w,:));





