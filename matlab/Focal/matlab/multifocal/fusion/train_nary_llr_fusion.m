function [alpha,beta] = train_nary_llr_fusion(scores,classf,epsilon,lambda,prior,alpha0,beta0,quiet);
%  TRAIN_NARY_LLR_FUSION: Train linear logistic regression fusion and calibration of K 
%                         input systems, to recognize N classes, over T supervised trials.
%                         If K=1, then pure calibration results.
%
%  usage:
%   [alpha,beta] = TRAIN_NARY_LLR_FUSION(scores,classf,epsilon,lambda,prior,alpha0,beta0);
%   [alpha,beta] = TRAIN_NARY_LLR_FUSION(scores,classf,epsilon,lambda,[],alpha0,beta0);
%   [alpha,beta] = TRAIN_NARY_LLR_FUSION(scores,classf,epsilon,lambda,prior);
%   [alpha,beta] = TRAIN_NARY_LLR_FUSION(scores,classf,epsilon,lambda);
%   [alpha,beta] = TRAIN_NARY_LLR_FUSION(scores,classf);
%
%   alpha = TRAIN_NARY_LLR_FUSION(scores,classf,epsilon,[],prior,alpha0);
%   alpha = TRAIN_NARY_LLR_FUSION(scores,classf,epsilon,[],prior);
%   alpha = TRAIN_NARY_LLR_FUSION(scores,classf,epsilon);
%   alpha = TRAIN_NARY_LLR_FUSION(scores,classf);
%
%  Input parameters:
%      scores  : 1-by-K cell array, where each element is an N by T matrix. K >= 1
%                These are N scores for each of T trials for each of K input systems: 
%                  scores{k}(i,t) is the score of system k, for recognizing class i in trial t
%                Scores have log-likelihood nature: 
%                  more positive score favours class i, 
%                  more negative score favours all the other classes.
%
%      classf  : 1-by-T row, with elements in {1,2, ..., N}.
%                These are the true class labels
%
%      epsilon : positive regularization constant for alpha coefficients 
%                optional: default = 0 
%
%      lambda  : positive regularization constant for beta coefficients
%                optional: default = 1.0e-6 
%
%      prior   : N-component column vector; prior distribution
%                optional: default = ones(N,1)/N
%
%      alpha0  : K-component column vector; initialization for alpha
%                optional: default = zeros(K,1)
%
%      beta0   : N-component column vector; initialization for beta
%                optional: default = zeros(N,1)
%
%      quiet   : 0 prints optimization progress information for every iteration
%                1 prints only warnings and errors
%                optional: default = 0
%
%  Output parameters:
%      alpha  : K-component column vector; system combination weights
%
%      beta   : N-component column vector; score offsets
%               optional: default = zeros(K,1)
%
%
%  See also:  APPLY_NARY_FUSION to calibrate and/or fuse new data.



%  This code is a much adapted version of the m-file 'train_cg.m' as made available by Tom Minka
%  at http://www.stat.cmu.edu/~minka/papers/logreg/.

n_iter = 300;

if nargin < 8
   quiet = 0;
end;

if ~iscell(scores)
   scores = {scores};
end;


K0 = length(scores); % number of input systems
[N,T] = size(scores{1});
if ~quiet
   if K0>1
      fprintf('\nTraining calibrated fusion of %i input systems; %i classes; %i trials\n',K0,N,T);
   else
      fprintf('\nCalibrating 1 input system; %i classes; %i trials\n',N,T);
   end;
end;


classf = classf(:)'; %make sure classf is row vector
if (min(classf)~=1 | max(classf)~=N | length(classf)~=T),
   error('illegal parameter classf');
end;
lookup = classf+[0:T-1]*N;  

% if beta is required, (implicitly) add N dummy systems
if (nargout>1),
   if ~quiet
      fprintf('offsets (beta) will be trained\n');
   end;
   K = K0+N;
else
   if ~quiet
      fprintf('offsets (beta) not trained: assumed to remain zero\n');
   end;   
   K = K0;
end;

if (nargin<5) | isempty(prior),
   prior = ones(N,1)/N;
   if ~quiet
      fprintf('using default flat prior\n');
   end;   
else
   prior = prior(:);
end;

counts = zeros(1,N);
for i=1:N;
   f=find(classf==i);
   counts(i) = length(f);
end;
if ~quiet
   fprintf('class counts: %s\n',num2str(counts));
end;   
if (min(counts==0)),
   error('zero counts for one or more classes');
end;


weights = zeros(1,T);
for t=1:T;
   i = classf(t);
   weights(t) = prior(i)/counts(i);
end;


alpha = zeros(K,1);
if (nargin>5) & ~isempty(alpha0)
   alpha(1:K0) = alpha0(:);
end;
if (K>K0) & (nargin>6) & ~isempty(beta0)
   alpha(K0+1:K) = beta0(:);
end;

if (nargin<3) | isempty(epsilon)
   epsilon = 0;
end;

if (nargin<4) | isempty(lambda)
   lambda = 1.0e-6;
end;


reg_weights = [epsilon*ones(K0,1)];
if (K>K0),
   reg_weights = [reg_weights;lambda*ones(N,1)];
end;

log_prior = log(prior);%-mean(log(prior));

g_const = zeros(K,1);
for k=1:K0
  s = scores{k}; 
  g_const(k) = s(lookup)*weights';  % s(lookup) gives score for true class   
end;
for k=K0+1:K % dummies
   i = k-K0;
   g_const(k) = sum(weights(find(classf==i)));   
end;


g = zeros(K,1);
old_g = zeros(size(g));
old_alpha = zeros(size(alpha));
ss = zeros(K0,T);     
old_obj = inf;
step = 1;
for iter = 1:n_iter
   
   
  backtrack = 1;
  while backtrack 
  
     % arg = Logit posterior, one column per trial 
     arg = alpha(1)*scores{1};
     for k=2:K0;
        arg = arg + alpha(k)*scores{k};
     end;
     for k=K0+1:K; % betas + log(prior)
        i = k-K0;
        arg (i,:) = arg (i,:) + (alpha(k)+log_prior(i));
     end;
  
     lsm = logsoftmax(arg);
     sigma = exp(lsm); % posterior
  
     % objective in normalized cost form
     obj = -lsm(lookup)*weights'+0.5*(alpha.^2)'*reg_weights; % convex cost (=log(2)Cllr if prior is flat)
     obj = obj /log(N); 
  
     if obj > old_obj
        step = step/10;
        alpha = old_alpha+step*u;
        backtrack = 1;
        if ~quiet
           fprintf('iter %i: obj=%g, backtracking \n',iter,obj);
        end;   
     else
        backtrack=0;
     end;
  end;
  
  % convergence test (here obj < old_obj)
  gradient_too_small = (iter>1) & ( sqrt(g'*g) < 1.0e-8 );
  gradient_small = (iter>1) & ( sqrt(g'*g) < 1.0e-3 );
  movement_small = (iter>1) & (max(abs(alpha - old_alpha)) < 1e-5);
  improvement_small = (iter > 1) & (obj <= old_obj) & (old_obj-obj<1.0e-5);
  tired = iter>5;
  if gradient_too_small | ( tired & gradient_small & movement_small & improvement_small ) 
     if ~quiet, 
        fprintf('--convergence criterium satisfied--\n'); 
     end;
     if (K>K0)
       beta = alpha(K0+1:K);
     end;
     alpha = alpha(1:K0);
     return;
  end;

  old_obj = obj;
  
  % g = gradient (Minka's objective is concave likelihood, i.e. negative of  convex cost)
  g = g_const - (reg_weights.*alpha);
  for k=1:K0
    s = scores{k}; 
    ss(k,:) = sum(sigma.*s,1);
    g(k) = g(k) - ss(k,:)*weights';  
  end;
  for k=K0+1:K
    i = k-K0; 
    g(k) = g(k) - sigma(i,:)*weights';  
  end;
  
  % choose search direction u
  if iter == 1
     u = g;
     ic = 0;
  else
    [u,ic] = rcg_dir(u, g, old_g,ic,quiet);
  end
  
  
  %compute u'*H*u = (2nd derivative along u), where H is Hessian
  
  su = u(1)*scores{1};
  for k=2:K0
     su = su + u(k)*scores{k};
  end;
  for k=K0+1:K; % dummies
     i = k-K0;
     su(i,:) = su(i,:) + u(k);
  end;
  
  uHu = -u'*(reg_weights.*u);
  uHu = uHu - sum(sigma.*(su.^2),1)*weights';
  if (K>K0)
     uHu = uHu + ((u(1:K0)'*ss+u(1+K0:K)'*sigma).^2)*weights';
  else
     uHu = uHu + ((u(1:K0)'*ss).^2)*weights';
  end;
  
  % single Newton-step line-search along u, 
  %   using 1st (u'g) and 2nd (u'Hu) derivatives along u to find step size
  ug = u'*g;
  step = -ug/uHu;
  old_alpha = alpha;
  alpha = alpha + step*u;    
  
  old_g = g;
  %if max(abs(alpha - old_alpha)) < 1e-5
    %break;
  %end
  
  if ~quiet
     fprintf('iter %i: obj=%g, u''Hu=%g, |g|=%g, |u|=%g \n',iter,obj,uHu,sqrt(g'*g),sqrt(u'*u));
  end;   
end
if iter == n_iter
   fprintf('Convergence criterion not met within %i iterations\n',n_iter);
   fprintf('If better convergence is desired, re-run with alpha0=alpha (and beta0=beta).\n');
end

if (K>K0)
   beta = alpha(K0+1:K);
end;
alpha = alpha(1:K0);

