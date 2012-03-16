function data_out = apply_hlda(Trans,data_in);
% APPLY_HLDA: Applies HLDA transformation: 
%
%    data_out = Trans*data_in
%
%  Usage: data_out = apply_hlda(Trans,data_in);
%
%  Also see: TRAIN_HLDA()

data_out = Trans*data_in;