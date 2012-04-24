close all
addpath(genpath('../../Focal'))


[trainCellScores, trainSuperScores, trainlabels] = readScores('../../train.txt');
[devCellScores, devSuperScores, devlabels] = readScores('../../dev.txt');
[testCellScores, testSuperScores, testlabels] = readScores('../../nist.txt');

priors = ones(1, 12)/12;%flat prior

%Closed set
temp = [testlabels ~= 0];
cTestCellScores = {testCellScores{1}(:, temp), testCellScores{2}(:, temp)};%closed set test scores
cTestSuperScores = testSuperScores(:, temp);
cTestLabels = testlabels(temp);%closed set test labels


%llr fusion
[alpha, beta] = train_nary_llr_fusion(devCellScores, devlabels);
llrFuseScores = apply_nary_lin_fusion(cTestCellScores, alpha, beta);

%linear backend
[Trans, offset] = train_linear_backend(devSuperScores, devlabels, {'ppca', 11});
linScores = apply_linear_backend(cTestSuperScores, Trans, offset);

%Kjører linear backend igjen for  å få ned refinement
temp = apply_linear_backend(devSuperScores, Trans, offset);
[alpha, beta] = train_nary_llr_fusion(temp, devlabels);
llrLinFuseScores = apply_nary_lin_fusion(linScores, alpha, beta);

%quadratic backend
[CC, Mu] = train_quadratic_backend(devSuperScores, devlabels, {'ppca', 11}, 0.9);
quadScores = apply_quadratic_backend(cTestSuperScores, CC, Mu);


calref_plot({cTestCellScores{1}, cTestCellScores{2}, llrFuseScores, linScores, llrLinFuseScores, quadScores}, cTestLabels, {'Prim dialect', 'sec dialect', 'llrfuseScores', 'linScores', 'llr+lin fuse', 'quadScores'});