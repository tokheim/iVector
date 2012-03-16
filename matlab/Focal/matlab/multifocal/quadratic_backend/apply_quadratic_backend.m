function loglh = apply_quadratic_backend(data,CC,Mu);
% APPLY_QUADRATIC_BACKEND: applies quadratic (heteroscedastic gaussian) backend to new data.
%
% Usage: loglh = APPLY_QUADRATIC_BACKEND(data,CC,Mu);
%
% Input parameters: 
%
%      data  : D by T array of T data points of dimension D.
%
%      CC,Mu : parameters of backend as 
%              trained by TRAIN_QUADRATIC_BACKEND().
%
% Output parameters:
%
%      loglh: N by T matrix of relative log-likelihoods, 
%             (log-likelihoods for each of N classes, given each of T trials)
%
%             loglh has the following interpretation:
%
%               loglh(i,t) = log P(trail_t | class_i) - offset_t, 
%
%             where:
%               log denotes natural logarithm
%               offset_t is an unspecified real constant that may vary by trial




N = length(CC);
[D,T] = size(data);


loglh = zeros(N,T);
for i=1:N;
   [R,p] = chol(CC{i});
   if p~=0
      warning('singular covariance, outputting zeros');
      loglh = zeros(N,T);
      return;
   else
     L = R';   
   end;
   logdet = 2*sum(log(diag(L)));
   invL_Mu = L\Mu(:,i);
   k = -0.5*(logdet+invL_Mu'*invL_Mu);
   invL_X = L\data;
   loglh(i,:) = k-0.5*sum(invL_X .^2,1)+invL_Mu'*invL_X;
end;
