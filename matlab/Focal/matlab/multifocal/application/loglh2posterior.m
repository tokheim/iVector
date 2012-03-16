function post = loglh2posterior(loglh,prior);
%  LOGLH2POSTERIOR: log-likelihood to posterior transformation (Bayes' rule).
%  
%
%  Usage:
%
%    posterior = LOGLH2POSTERIOR(loglh,prior)
%
%  Input parameters:
%       loglh  : N by T matrix of relative log-likelihoods, 
%                There are N log-likelihoods for each of T trials:
%                  loglh(i,t) = log P(trail_t | class_i) - offset_t, 
%                where:
%                   log denotes natural logarithm
%                   offset_t is an unspecified real constant that may vary by trial
%       prior  : N-component prior probability distribution (non-negative, components sum to one).
%
%  Output:
%       post: N by T matrix: T column vectors, each being a posterior probability distribution
%                            for the N classes, given the associated trial.

[N,T] = size(loglh);
prior = prior(:);
if length(prior) ~= N
   error('loglh and prior incompatible');
end;
if min(prior) <0
   error('illegal prior: has negative component(s)');
end;
if abs(log(sum(prior))) > 0.001
   error('illegal prior: does not sum to one');
end;

post = softmax(loglh+repmat(log(prior),1,T));