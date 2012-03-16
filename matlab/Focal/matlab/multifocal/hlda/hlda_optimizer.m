function A = hlda_optimizer(A, p, SIGMA_g, SIGMA, gamma, iters, initers, alpha)
%HLDA Heteroscedastic linear discriminant analysis
%   NewA = hlda(A, p, SIGMA_g, SIGMA, gamma, iters) performe
%   iterative optimization of HLDA transformation matrix and
%   return its new estimate NewA. Base vectors are stored column-wise
%   in matrices A and NewA. For dimensionality reduction, use only
%   first p columns of the final transformation. After each iteration,
%   value of HLDA objective function is printed. The increase of
%   this value is directly related to the increase of the likelihood
%   obtained by use of the "improved" transformation.
%
%   A        - initial guess of transform matrix
%              (e.g. eye(size(SIGMA_g)) or LDA trasformation matrix)
%   p        - number of useful (wanted) dimensions
%   SIGMA_g  - global covariance matrix
%   SIGMA    - cell array of class covarience matrices
%   gamma    - vector of class occupancy counts (samples per class)
%   iters    - number of optimization iterations
%
%   Optional arguments:
%   initers  -  number of optimization inner loop iterations
%   alpha    - momentum (value between 0 and 1, default is 1)
%
%     See also COV
%
%   References
%
%     N. Kumar, Investigation of Silicon-Auditory Models and
%     Generalization of Linear Discriminant Analysis for Improved
%     Speech Recognition, Ph.D. Thesis, John Hopkins University,
%     Baltimore, 1997
%
%     M.J.F. Gales, Semi-tied covariance matrices for hidden Markov
%     Models", IEEE Transaction Speech and Audio Processing, vol. 7,
%     pp. 272-281, 1999.

% Author: Lukas Burget, 31 July 2006.
% Modified by Niko Brummer, 23 May 2007, to run faster for higher dimensions, 
%                                        by using matrix inversion lemma.        

if nargin < 8
  alpha = 1;
end

if nargin < 7
  initers = 10;
end


if nargin < 6
   iters = 300;
   quiet = 1;
else
   quiet = 0;
end;


M = length(SIGMA);    % number of classes
d = size(SIGMA_g, 1); % original feature space dimension
tau = sum(gamma);     % total nuber of samples
momentum=zeros(d, d);

if p > d
   error('illegal parameter p: must be <= data dimension');
end;

if det(A) < 1,
  disp('Init guess matrix does not have positive determinant. Multiplying first column by -1');
  A(:,1) = A(:,1) * -1;
end

if ~quiet
   disp(' ')
   disp('Iteration    Q_hlda(M,M'')')
   disp('-------------------------')
end;


Id = eye(d);
invSIGMA_g = Id/SIGMA_g;
%invSIGMA_g = inv(SIGMA_g);
Q = -inf;
for iter =1:iters,
   
  % Normalize determinant for better numerical precision 
  A = A / (det(A)^(1/d));

  %Q = tau * log(det(A')^2); %- tau * d * (log(2*pi)+1);
  oldQ = Q;
  Q = 0;
  for i =1:d,
    if i <= p
      G = zeros(d, d);
      for m = 1:M,
        sigma_i = A(:,i)' * SIGMA{m} * A(:,i);
        G = G + gamma(m) / sigma_i * SIGMA{m};
        Q = Q - gamma(m) * log(sigma_i);
        %invG{i} = inv(G);
        invG{i} = G\Id;
      end
    else
      sigma_i = A(:,i)' * SIGMA_g * A(:,i);
      %G = tau / sigma_i * SIGMA_g;
      Q = Q - tau * log(sigma_i);
      invG{i} = (sigma_i/tau)*invSIGMA_g;
    end
  end

  Q = Q / 2;
  if ~quiet
     fprintf('%5d        %.6g\n', iter-1, Q);
  end;   
  
  
  if (iter>20) & abs((Q-oldQ)/Q)<1.0e-6
     if ~quiet
        fprintf('convergence criterium satisfied\n');
        break;
     end;   
  end;
  
  
  for initer =1:initers,
     startA = A;
     % best to refresh invA and detA, for numerical accuracy
     invA = inv(A); 
     if initer==1, detA = 1; else detA = det(A); end;
     for i =1:d,
        C = invA*detA;
        ci_invG = C(i,:) * invG{i};
        oldCol = A(:,i);
        newCol = (ci_invG * sqrt(tau / (ci_invG * C(i,:)')))';
        A(:,i) = newCol;
        if i<d,
           [invA,detA] = inv_det_col_update(invA,detA,i,oldCol,newCol);
        end;
     end
     momentum = (1-alpha) * momentum + alpha * (A - startA);
     A = startA + momentum;
  end
end

% Normalize final transformation to have determinant equal to 1 
% i.e. A is a volume preserving transformation
A = A / (det(A)^(1/d));
