function [Trans,CC,Mu] = train_hlda_backend(data,classf,hlda_dim,smoothing,iters);
% TRAIN_HLDA_BACKEND Train a quadratic gaussian backend, preceded by an HLDA transformation.
%                    The HLDA model supposes that the input space can be partitioned into
%                    two linear subspaces: 
%                      - Subspace A, of dimension data_dim - hlda_dim,
%                        where the means and covariances accross classes are identical.
%                      - Subspace B, of dimension hlda_dim,
%                        where the means and covariances accross classes are different, 
%                        and where the covariances are diagonal.
%                    This code finds the optimal subspaces with an ML criterium and returns the
%                    HLDA transformation matrix, as well as the diagonal class  
%                    covariances and class means in the subspace B.
%                  
%
%  Usage: [Trans,CC,Mu] = TRAIN_HLDA_BACKEND(data,classf,hlda_dim,smoothing,iters)
%         [Trans,CC,Mu] = TRAIN_HLDA_BACKEND(data,classf)
%
%  Input parameters:
%    data: a D by T array, where each of T data points is a D-dimensional column vector
%
%    classf: 1 by T row-vector, with elements ranging from 1..N.
%                These are the true class labels for every data point.
%
%    hlda_dim: HLDA output dimension, 
%              range: 1 <= dim =< D, where D is input dimension
%              (optional, default = N-1, where N = no. of classes)
%
%    smoothing: smoothing factor for class-covariance estimates. 
%               (optional, default is no smoothing)
%               See CLASS_COVARIANCES(...,smoothing) for more info. 
%
%    iters: max number of HLDA optimization iterations.
%           (optional, iterations are not displayed if not specified)
%           See HLDA_OPTIMIZER(...,iters,...) for more info. 
%
%  Output parameters:
%
%      Trans: An hlda_dim-by-D matrix to project (and rotate) D-dimensional inputs to 
%            hlda_dim-dimensional outputs.
%      CC: 1-by-N cell array of hlda_dim-by-hlda_dim class covariances in transformed space.
%      Mu: hlda_dim-by-N matrix of class means in transformed space.



D = size(data,1);
N = max(classf);

if nargin<3
   hlda_dim = N-1;
end;

if nargin<4
   smoothing = 0;
end;


[CC,counts,Mu] = class_covariances(data,classf,{},smoothing);
GC = ml_covariance(data);
A = eye(size(data,1));
if nargin<5
   A = hlda_optimizer(A,hlda_dim,GC,CC,counts);
else
   A = hlda_optimizer(A,hlda_dim,GC,CC,counts,iters);
end;
Trans = A(:,1:hlda_dim)';

for i=1:N;
   CC{i} = diag(diag(Trans*CC{i}*Trans'));
end;
Mu = Trans*Mu;



