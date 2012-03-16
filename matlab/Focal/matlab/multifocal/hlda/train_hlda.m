function Trans = train_hlda(data,classf,hlda_dim,smoothing,num_iters);
% TRAIN_HLDA: Train an HLDA transformation.
%                    The HLDA model supposes that the input space can be partitioned into
%                    two linear subspaces: 
%                      - Subspace A, of dimension data_dim - hlda_dim,
%                        where the means and covariances accross classes are identical.
%                      - Subspace B, of dimension hlda_dim,
%                        where the means and covariances accross classes are different, 
%                        and where the covariances are diagonal.
%                    This code finds the optimal subspaces with an ML criterium and returns the
%                    HLDA projection matrix (onto subspace B).
%                  
%
%  Usage: Trans = TRAIN_HLDA_BACKEND(data,classf)
%         Trans = TRAIN_HLDA_BACKEND(data,classf,hlda_dim)
%         Trans = TRAIN_HLDA_BACKEND(data,classf,hlda_dim,smoothing)
%         Trans = TRAIN_HLDA_BACKEND(data,classf,hlda_dim,smoothing,iters)
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
%    smoothing: back-off smoothing factor for class-covariance estimates. 
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



if nargin<4
   smoothing = 0;
end;


[CC,counts] = class_covariances(data,classf,{},smoothing);

GC = ml_covariance(data);

A = eye(size(data,1));
if nargin<5
   A = hlda_optimizer(A,hlda_dim,GC,CC,counts);
else
   A = hlda_optimizer(A,hlda_dim,GC,CC,counts,num_iters);
end;

Trans = A(:,1:hlda_dim)';
