function [W,Psi] = factor_analysis_em(X,m,num_iter);
% Maximum Likelihood Factor Analysis using the EM-algorithm. 
%
% Reference: Z. Ghahramani and G. E. Hinton, 
%            "The EM algorithm for mixtures of factor analyzers," 
%            Tech. Report CRG-TR-96-1, Univ. of Toronto, 1997.
%            Online: www.cs.toronto.edu/~hinton/absps/tr-96-1.pdf 
%
% FA Model: x = Wz + mu + eps, where: 
%                                    x is d-dimensional observed vector, 
%                                    z is m-dimensional hidden vector, ~N(0,I)
%                                    W is factor loading matrix,
%                                    mu is origin
%                                    eps is noise ~N(0,Psi), where Psi is diagonal
%
% Usage: [W,Psi] = factor_analysis_em(X,m,num_iter)
% Usage: [W,Psi] = factor_analysis_em(X,m)
%
% Input:
%        X: d by n input array (observed data: n points of dimension d)
%        m: number of hidden factors, m <= d
%        num_iter: max no of EM iterations
%                  (optional: default = 100)
%                  If num_iter is specified, iteration progress is printed, otherwise quiet.
%
% Output:
%         W: d by m 'factor loading matrix'
%         Psi: d-dimensional vector of noise variances
%         mu: d-dimensional origin vector
%



if nargin<3
   num_iter = 300;
   quiet = 1;
else
   quiet = 0;
end;


[d,n]=size(X);
mu = mean(X')';
X = X-repmat(mu,1,n);
V = sum(X.^2,2)/n; 
X = X.*repmat(1./sqrt(V),1,n);

Psi0 = sum((X').^2)'; 

%Psi = ones(d,1); 
%W = randn(d,m);
[C,mu,sigma,W] = ppca_covariance(X,m);
Psi = sigma*ones(d,1); 

WtinvPsi = zeros(m,d);
for k=1:num_iter;
   
   oldPsi = Psi;
   oldW = W;
   
   WtinvPsi=(W.*repmat(Psi.^-1,1,m))';
   WtinvPsiW = WtinvPsi*W;
   Beta = WtinvPsi-WtinvPsiW*((eye(m)+WtinvPsiW)\WtinvPsi); 
   Ezx = Beta*X;   
   Ezzx = n*(eye(m)-Beta*W)+Ezx*Ezx';
   if (d>n) % select order of matrix multiplications to minimize work
      W = X*(Ezx'/Ezzx);
   else % d<=n
      W = (X*Ezx')/Ezzx;
   end;  
   temp = W*Ezx;
   Psi = (Psi0-sum(X'.*temp')')/n;
   full_trace = sum(Psi0)/n;
   movement = sum(sum((W-oldW).^2))+sum((sqrt(Psi)-sqrt(oldPsi)).^2);
   model_trace = sum(sum(W.^2))+sum(Psi);
   ratio = model_trace/full_trace;
   if ~quiet
      fprintf('%i: trace-ratio = %f; movement = %f\n',k,ratio,movement/full_trace);
   end;
   
   if (k>10) & (movement/full_trace < 1.0e-6)
      if ~quiet
         fprintf('convergence criterion satisfied\n');
      end;
      break;
   end;
end;

Psi = Psi.*V;
W = diag(sqrt(V))*W;