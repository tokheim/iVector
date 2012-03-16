function ll_out = perturb(scaling,offset,ll_in);
%PERTURB scales and translates log-likelihood vectors with the given parameters.
%
% ll_out = PERTURB(scaling,offset,ll_in);
% 
% INPUT parameters:
%   scaling: a positive real number to deflate log-likelihood magnitudes.
%   offset: a real column vector of M offsets by which the M class log-likelihoods are shifted.
%   ll_in: an M-by-N matrix of class-log-likelihoods for N data points.
%
% OUTPUT parameters:
%   ll_out: These likelihoods are perturbed thus:
%     ll_out = (ll_in - repmat(offset,1,N))/scaling

[m,n]=size(ll_in);
if (m ~= size(offset,1) )
   error('sizes of ll_in and offset disagree');
end;   
ll_out = (ll_in-repmat(offset,1,n))/scaling;

