function c = cdet(decisions,target);
% CDET: Auxiliary function for AVG_CDET_AND_CLLR

Cmiss = 1;
Cfa = 1;
if target
   c = Cmiss*(decisions==0);
else % non-target
   c = Cfa*(decisions==1);
end;   