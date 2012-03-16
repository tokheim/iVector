function c = avg_detection_cost(cost,evaluand,classf,open_set);
% CDET: Averages given cost function over 
%       target classes, using class priors as specified in NIST LRE-07 Eval-plan.
%
%   Usage: c = AVG_DETECTION_COST(cost,evaluand,classf,open_set);
%
%   Input parameters: 
%
%     cost: A string, which names the cost function for evaluatin every detection trial.
%           Use 'cdet' for average Cdet, or 'surprisal' for average Cllr.
%
%     evaluand: a NoTargets-by-T matrix of
%               0/1 decisions for CDET, or
%               log-likelihood-ratio scores for Cllr
%
%     classf : a row of T integers in the range 1,2,...,NoClasses.
%              Indicates the true class of each trial.
%              NoClasses = NoTargets for closed-set,
%              NoClasses = NoTargets +1 for open-set.
%
%     open_set: flag to select closed-set or open-set conditions:
%               0: closed-set. 
%               1: open-set. 
%               (optional, default = 0)  


if nargin<4
   open_set = 0;
end;

Ptar = 0.5;
Poos = 0.2;

[NoTargets,T] = size(evaluand);
NoClasses = max(classf);

if open_set
   if NoClasses ~= NoTargets+1 | min(classf)~=1 | length(classf) ~= T,
      error('illegal parameter classf');
   end;
else % closed-set
   if NoClasses ~= NoTargets | min(classf)~=1 | length(classf) ~= T,
      error('illegal parameter classf');
   end;
end;


% Priors(:,i) is prior distribution over classes, when target is class i
if open_set
   Priors = lre_priors(NoTargets,Ptar,Poos);
else % closed-set
   Priors = lre_priors(NoTargets,Ptar);
end;

c = 0;
for j=1:NoClasses;
   f_j = find(classf==j);
   count_j = length(f_j);
  for i=1:NoTargets;  
     c = c + Priors(j,i)*sum(feval(cost,evaluand(i,f_j),i==j))/count_j;
  end;
end;
c = c/NoTargets;