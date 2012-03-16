function Trans = train_lda(data,classf,regularization);
% TRAIN_LDA: Train an LDA to project from D-dimensional feature space to an 
%            (N-1)-dimensional linear subspace, which is discriminative under 
%            a homoscedastic Gaussian data modeling assumption.
%            Uses optional optional PPCA or FA regularization 
%            when estimating common-within-class covariance from data.
%
%   Regularization:
%      PPCA = probabilistic principal component analysis
%      FA = factor analysis
%
%   Note: This function is not presently used by any other part of this toolkit but is
%         included to make clear the relationship between the linear Gaussian
%         backend and LDA: The linear backend implicitly does do an LDA, 
%         but there are two differences: 
%         1. The LDA is purely linear, while the backend is affine---the backend 
%            additionally applies an offset for proper log-likelihood calibration. 
%         2. - The backend outpout is in symmetrical, but redundant log-likelihood format,
%              having N log-likelihoods, one for each of N classes.
%            - The LDA output is in asymmetrical non-redundant log-likelihood-ratio format,
%              having N-1 log-likelihood-ratios (and the offset is ignored).
%
%
%   Usage: Trans = TRAIN_LDA(data,classf,regularization);
%
%   Input parameters: 
%
%      data    : a D by T array, where each of T data points is an D-dimensional column vector.
%
%      classf  : 1 by T row-vector, with elements ranging from 1..N, where there are N classes.
%                These are the true class labels for every data point.
%
%      regularization: (optional                                                             ), 
%                      (default = no regularization i.e. unconstained ML covariance estimate)
%                      cell array, can be:
%                        {'ppca',rank}, where rank is the PPCA rank. See PPCA_COVARIANCE().
%
%                        {'fa',rank}, where rank is the FA rank. See FA_COVARIANCE().
%   Output parameters: 
%      
%      Trans: An (N-1)-by-D matrix to project D-dimensional inputs to (N-1)-dimensional outputs.
%             To apply this backend to an input vector x, do: 
%                y = Trans*x


[Trans,offset] = train_linear_backend(data,classf,regularization);
N = size(Trans,1);
Trans = [eye(N-1),-ones(N-1,1)]*Trans;