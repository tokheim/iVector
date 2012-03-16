function [C,sigma,W] = ppca(Cml,r);
% PPCA: Probabilistic Princinpal Component Analysis of Covariance matrix.
%
%
%    PPCA Modeling assumption: Covariance  = sigma I + W*W', where 
%                              I is D-by-D identity and
%                              W is D-by-r, with 0<= r <= d.  
%
%    This code computes the maximum-likelihood (ML) PPCA model, given the constraint
%    represented by the rank r, and given the data as represented by Cml, the
%    unconstrained ML covariance estimate for the data.
%
%    Usage:            C = PPCA_COVARIANCE(Cml,r);
%            [C,sigma,W] = PPCA_COVARIANCE(Cml,r);
%    
%    Input parameters: 
%
%      Cml : unconstrained D-by-D maximum-likelihood covariance
%
%      r   : an integer, where 0 <= r =< D.
%            Special cases: r = 0 gives isotropic covariance C = sigma I.
%                           r = D gives unconstrained covariance C = Cml, 
%                                 with sigma and W undefined.
%                           r = D-1 gives unconstrained covariance C = Cml, 
%                                   with sigma and W such that C = sigma I + W*W'.
%            The smaller r is made, the stronger the constraint and therefore the 
%            'regularization' effect. If the data is scarce, then choosing a small 
%            r is likely to be beneficial for using C for recognition purposes.
%
%
%    Output parameters:
%
%      C: PPCA-regularized covariance matrix.
%         Symmetric, D by D, positive definite (or semi-definite) matrix.
%         (If there are zero (or very small) eigenvalues, try to decrease r.)
%
%      sigma,W: (optional), model parameters, where C = sigma I + W*W'.

D = size(Cml,1);

if r==0 % just return isotropic covariance estimate
   W = [];
   sigma = trace(Cml)/D;
   C = sigma*eye(D);
   return;
end;


if r==D % just return full ml covariance
   sigma = 0;
   W = [];
   C = Cml;
   return;
end;



if (r<1) | (r>=D)
   error('illegal parameter r');
end;


[V,E] = eig(Cml);
e = diag(E);
[dummy,ii] = sort(e);
ii = ii(D:-1:1+D-r);
V = V(:,ii);
e = e(ii);
E = diag(e);
C0 = V*E*V';
sigma = (trace(Cml)-trace(C0))/(D-r);
W = V*diag(sqrt(e-sigma));
C = sigma*eye(D)+W*W';
C = (C+C')/2;
