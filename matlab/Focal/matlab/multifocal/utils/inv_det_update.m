function [newInv,newDet] = inv_det_update(invA,detA,u,v);
% INV_DET_UPDATE: Updates inverse and determinant of A+u'v, where u and v are vectors.
%

% Sherman Morrison formula
% See e.g.: http://http://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
lambda = v'*invA*u;
newInv = invA - (invA*u)*(v'*invA)/(1+lambda); 


% See: Brookes, M., "The Matrix Reference Manual", 2005 [online] 
%      www.ee.ic.ac.uk/hp/staff/dmb/matrix/intro.html
newDet = (1+lambda)*detA; 