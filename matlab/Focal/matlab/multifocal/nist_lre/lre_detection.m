function [llrs,decisions] = lre_detection(loglh,open_set);
% LRE_DETECTION: Converts from log-likelihood format to detection scores and decisions, 
%                suitable for use in the NIST-LRE-07. 
%
%                This function is applicable to both closed-set and open-set conditions.
%                In both cases, the prior distributions over language classes, as specified by
%                the NIST LRE-07 plan are used. 
%                See www.nist.gov/speech/tests/lang/2007/LRE07EvalPlan-v7e.pdf
%
%   Usage: [llrs,decisions] = LRE_DETECTION(loglh,open_set);
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
%       open_set: flag to select closed-set or open-set conditions:
%                 0: closed-set. In this case N = no. of target languages
%                 1: open-set. In this case N = no. of target languages + 1, where
%                    the last row of loglh gives the log-likelihoods for the out-of-set
%                    hypothesis.
%                 (optional, default = 0)  
%
%   Output parameters:
%     
%      llrs: K by T matrix of detection log-likelihood-ratios, where there are T trials
%            and where for closed-set K = N and for open-set K = N+1, where N is the 
%            column size of the input parameter loglh. 
%            In other words, K is the number of target languages and row llrs(i,:)
%            gives the detection log-likelihood-ratio for target language (or dialect) i.
%            These detection log-likelihood-ratios are suitabel for evaluation by the Cllr
%            metric specified in the evaluation plan.
%
%      decisions: K by T matrix of detection decisions, one row of decisions per target class.
%                 These decisions are made by thresholding the log-likelihood-ratios of llrs.
%                 The threshold (zero) is derived from the costs and priors specified in 
%                 the evaluatiopn plan. 
%                 A decision of '0' denotes reject and '1' denotes accept.



% Parameters for priors and costs as specified in the evaluation plan:
Ptar =0.5;
Poos = 0.2;
Cfa = 1;
Cmiss = 1;

if nargin<2
   open_set = 0;
end;

[N,T] = size(loglh);

if open_set
   NoTargets = N-1;
   Priors = lre_priors(NoTargets,Ptar,Poos);
else %closed-set  
   NoTargets = N;
   Priors = lre_priors(NoTargets,Ptar);
end;

thresh = log(Cfa/Cmiss) - log(Ptar/(1-Ptar));
llrs = zeros(NoTargets,T);
decisions = zeros(NoTargets,T);
for i=1:NoTargets;
   llrs(i,:) = loglh2detection_llr(loglh,Priors(:,i),i);
   decisions(i,:) = llrs(i,:)>thresh;    
end;