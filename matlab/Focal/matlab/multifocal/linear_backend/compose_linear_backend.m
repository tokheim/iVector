function [Trans,offset] = compose_linear_backend(C,Mu);
% COMPOSE_LINEAR_BACKEND: Utility function, called by TRAIN_LINEAR_BACKEND().
%                         Derives an affine transform from D-dimensional feature space 
%                         N-dimensional log-likelihood space, under homoscedastic
%                         Gaussian data modeling assumption.
%                         (The multiplicative part of the transform may be considered to be 
%                         an LDA, see TRAIN_LDA().)
%
%     Input parameters:
%
%             Mu: D-by-N matrix of D-dimensional feature-space means for each of N classes.
%
%             C: D-by-D, common-within-class covariance. 
%                Must be invertible! Warning is issued if not and outputs are zeroed.
%
%     Output parameters:
%
%             Trans: N-by-D matrix, to transform from D-dimensional feature space to
%                    N-dimensional log-likelihood space.
%      offset: an N-dimensional offset vector, to be applied after projection.
%              To apply this backend to an input vector x, do: 
%                y = Trans*x+offset


[R,p]=chol(C);
if p~=0
   warning('singular covariance, outputting zeros');
   [D,N] = size(Mu);
   Trans = zeros(N,D);
   offset = zeros(N,1);
end;

% If documentation is to be believed, this is solved via Cholesky transform, which is
% faster and more accurate than using inv(C):
invC_Mu = C\Mu; 

Trans = invC_Mu'; 
offset = -0.5*sum(Mu.*invC_Mu)'; 

