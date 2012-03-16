function Priors = lre_priors(NoTargets,Ptar,Poos);
% LRE_PRIORS: Returns all of the prior distributions as needed in the 
%             closed-set or open-set conditions of the NIST LRE-07 language detection task.
%             See www.nist.gov/speech/tests/lang/2007/LRE07EvalPlan-v7e.pdf
%
%   Usage: Priors = LRE_PRIORS(NoTargets,Ptar); % for the closed-set condition
%          Priors = LRE_PRIORS(NoTargets,Ptar,Poos); % for the open-set condition
%
%   Input parameters:
%
%     NoTargets: The number of target languages or dialects, for the particular condition.
%
%     Ptar: The target prior. Use Ptar = 0.5 for LRE-07.
%
%     Poos: The prior for the out-of-set hypothesis. Use Ptar = 0.2 for LRE-07.
%           (optional, defaults to 0, which implies closed-set) 
%
%   Output parameters:
%
%     Priors: A NoClasses-by-NoTargets matrix, where the prior probability distribution
%             applicable to detection of target i, is the column Priors(:,i). 
%             For closed-set, NoClasses = NoTargets.
%             For open-set, NoClasses = NoTargets+1, where the last class is the 
%               out-of-set class.

if nargin<3 % closed-set
   N = NoTargets;
   Priors = (ones(N)-eye(N))*(1-Ptar)/(N-1) + eye(N)*Ptar;
else % open-set
   N = NoTargets+1;
   Priors = (ones(N-1)-eye(N-1))*(1-Ptar-Poos)/(N-2) + eye(N-1)*Ptar;
   Priors = [Priors;Poos*ones(1,N-1)];
end;