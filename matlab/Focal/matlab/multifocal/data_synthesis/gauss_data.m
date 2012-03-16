function [data,classf] = gauss_data(C,Mu,trialsPerCLass); 

T = trialsPerCLass;

[D,N] = size(Mu);
data = zeros(D,T*N);
classf = zeros(1,T*N);

if iscell(C) % heteroscedastic
   if length(C) ~= N
      error('illegal argument C, must be single matrix or cell-array of length N')
   end;
   range = 1:T;
   for i=1:N
      data(:,range) = chol(C{i})'*randn(D,T)+repmat(Mu(:,i),1,T);   
      classf(range) = i;
      range = range + T;
   end;
else % homoscedastic
   L = chol(C)';
   range = 1:T;
   for i=1:N
      data(:,range) = L*randn(D,T)+repmat(Mu(:,i),1,T);   
      classf(range) = i;
      range = range + T;
   end;
end;
