function [newInv,newDet] = inv_det_col_update(invA,detA,colIndx,oldCol,newCol);
% INV_DET_UPDATE: Updates inverse and determinant of A, afer modification of a column of A.


u = newCol-oldCol;
%v = zeros(length(u),1); v(colIndx) = 1; 


% Sherman-Morrison formula (also Woodbury formula, or Matrix inversion lemma).
% See e.g.: http://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
vTinvA = invA(colIndx,:); % v'*invA
lambda = vTinvA*u;
newInv = invA - (invA*u)*(vTinvA/(1+lambda)); 


% Matrix determinant lemma:
% See:
%   1. http://en.wikipedia.org/wiki/Matrix_determinant_lemma
%   2. Brookes, M., "The Matrix Reference Manual", 2005 [online] 
%      www.ee.ic.ac.uk/hp/staff/dmb/matrix/intro.html
newDet = (1+lambda)*detA; 