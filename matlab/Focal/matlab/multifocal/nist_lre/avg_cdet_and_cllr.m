function [avg_cdet,avg_cllr] = avg_cdet_and_cllr(loglh,classf,open_set);
% AVG_CDET_AND_CLLR: Does an LRE detection evaluation, with average Cdet and average Cllr.
%
%                The following steps are performed:
%                  1. Maps recognition log-likelihoods to detection log-likelihood-ratios.
%                  2. Evaluates detection log-likelihood-ratios via average Cllr over targets.
%                  3. Thresholds detection log-likelihood-ratios to make hard decisions.
%                  4. Evaluates decisions via average Cdet over targets. 
%
%   Usage: [avg_cdet,avg_cllr] = AVG_CDET_AND_CLLR(loglh,classf,open_set);
%
%   Input parameters:
%
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
%       open_set: flag to select closed-set or open-set conditions:
%                 0: closed-set. 
%                 1: open-set. 
%                 (optional, default = 0)  
%
%   Output:
%
%     avg_cdet: Average CDET as defined by the NIST LRE-07 eval plan.
%     
%     avg_cllr: Average detection Cllr (not multiclass Cllr), averaged in the same way
%               as average Cdet, but using logarithmic cost, instead of 
%               misclassification cost.



if nargin<3
   open_set = 0;
end;

[llrs,decisions] = lre_detection(loglh,open_set);
avg_cdet = avg_detection_cost('cdet',decisions,classf,open_set);
avg_cllr = avg_detection_cost('surprisal',llrs,classf,open_set);

