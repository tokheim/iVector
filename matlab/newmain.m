%%setup

close all
addpath(genpath('../../Focal'))

%trainLoc = '../../map200dev.txt';
%testLoc = '../../map200evl.txt';
%trainLoc = '../../nomap200resetDev.txt';
%testLoc = '../../nomap200resetEvl.txt';
trainLoc = '../../hamNomap200resetDev.txt';
testLoc = '../../hamNomap200resetEvl.txt';

%Baseline systems
%trainLoc = '../../mapBaseDev.txt';
%testLoc = '../../mapBaseEvl.txt';
%trainLoc = '../../nomapBaseDev.txt';%Worse performance
%testLoc = '../../nomapBaseEvl.txt';%Worse performance

%%Closed set testing

[devScores, devLabels] = readScores(trainLoc, 'ignore');
[testScores, testLabels] = readScores(testLoc, 'ignore');

[Trans, offset] = train_linear_backend(devScores, devLabels, {'ppca', 20});
linScores = apply_linear_backend(testScores, Trans, offset);

%[CC, Mu] = train_quadratic_backend(devScores, devLabels, {'ppca', 11}, 0.9);
%quadScores = apply_quadratic_backend(testScores, CC, Mu);

[llrs, decision] = lre_detection(linScores, 0);
cdet = avg_detection_cost('cdet', decision, testLabels, 0);

disp(['AVG cdet for closed set: ' num2str(cdet)]);

%%Open set testing
[devScores, devLabels] = readScores(trainLoc, 'UBM');
[testScores, testLabels] = readScores(testLoc, 'include');

[Trans, offset] = train_linear_backend(devScores, devLabels);
linScores = apply_linear_backend(testScores, Trans, offset);

%[CC, Mu] = train_quadratic_backend(devScores, devLabels, {'ppca', 11}, 0.9);
%quadScores = apply_quadratic_backend(testScores, CC, Mu);

[llrs, decision] = lre_detection(linScores, 1);
cdet = avg_detection_cost('cdet', decision, testLabels, 1);

disp(['AVG cdet for open set: ' num2str(cdet)]);

% [trainCellScores, trainSuperScores, trainlabels] = readScores('../../train.txt');
% [devCellScores, devSuperScores, devlabels] = readScores('../../dev.txt');
% [testCellScores, testSuperScores, testlabels] = readScores('../../nist.txt');
% 
% priors = ones(1, 12)/12;%flat prior
% 
% %Closed set
% temp = [testlabels ~= 0];
% cTestCellScores = {testCellScores{1}(:, temp), testCellScores{2}(:, temp)};%closed set test scores
% cTestSuperScores = testSuperScores(:, temp);
% cTestLabels = testlabels(temp);%closed set test labels
% 
% 
% %llr fusion
% [alpha, beta] = train_nary_llr_fusion(devCellScores, devlabels);
% llrFuseScores = apply_nary_lin_fusion(cTestCellScores, alpha, beta);
% 
% %linear backend
% [Trans, offset] = train_linear_backend(devSuperScores, devlabels, {'ppca', 11});
% linScores = apply_linear_backend(cTestSuperScores, Trans, offset);
% 
% %Kj�rer linear backend igjen for  � f� ned refinement
% temp = apply_linear_backend(devSuperScores, Trans, offset);
% [alpha, beta] = train_nary_llr_fusion(temp, devlabels);
% llrLinFuseScores = apply_nary_lin_fusion(linScores, alpha, beta);
% 
% %quadratic backend
% [CC, Mu] = train_quadratic_backend(devSuperScores, devlabels, {'ppca', 11}, 0.9);
% quadScores = apply_quadratic_backend(cTestSuperScores, CC, Mu);
% 
% 
% calref_plot({cTestCellScores{1}, cTestCellScores{2}, llrFuseScores, linScores, llrLinFuseScores, quadScores}, cTestLabels, {'Prim dialect', 'sec dialect', 'llrfuseScores', 'linScores', 'llr+lin fuse', 'quadScores'});