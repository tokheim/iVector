close all
addpath(genpath('multifocal'))


[trainscores, labels] = readScores('../train');
[devscores, labels] = readScores('../dev');
[testscores, labels] = readScores('../nist');

%train_nary_llr_fusion(devscores)
%