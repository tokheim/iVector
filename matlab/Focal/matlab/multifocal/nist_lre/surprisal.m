function c = surprisal(llr,target);
% SURPRISAL: Auxiliary function for AVG_CDET_AND_CLLR


if target
   c = neglogsigmoid(llr)/log(2);
else % non-target
   c = neglogsigmoid(-llr)/log(2);
end;   