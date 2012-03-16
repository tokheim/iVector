function y=logsumexp(x);
% 	y = logsumexp(x);
%    Mathematically the same as y=log(sum(exp(x))), 
%    but guards against numerical overflow of exp(x).
%
[m,n]=size(x);
xmax=max(x,[],1);
xnorm=x-repmat(xmax,m,1);
ex=exp(xnorm);
y=xmax+log(sum(ex,1));
