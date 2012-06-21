%%setup
disp('--NEW TEST--')
close all
addpath(genpath('../../Focal'))

trainLoc = '../../results/10/nomap150resetDev.txt';
testLoc = '../../results/10/nomap150resetEvl.txt';

%Baseline systems
%trainLoc = '../../results/3/mapBaseDev.txt';
%testLoc = '../../results/3/mapBaseEvl.txt';

%%Closed set testing

[devScores, devLabels] = readScores(trainLoc, 'ignore');
[testScores, testLabels] = readScores(testLoc, 'ignore');

[Trans, offset] = train_linear_backend(devScores, devLabels);
linScores = apply_linear_backend(testScores, Trans, offset);

%[CC, Mu] = train_quadratic_backend(devScores, devLabels, {'ppca', 11}, 0.9);
%quadScores = apply_quadratic_backend(testScores, CC, Mu);

[llrs, decision] = lre_detection(linScores, 0);
cdet = avg_detection_cost('cdet', decision, testLabels, 0);

disp(['AVG cdet for closed set: ' num2str(cdet)]);

figure(1)
h1 = plotdet(llrs, testLabels);

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

hold on
h2 = plotdet(llrs, testLabels, 'r--');
legend([h1(1) h2(1)], 'Closed set', 'Open set')