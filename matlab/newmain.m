close all
addpath(genpath('multifocal'))

temp = dlmread('train');

trainscores = temp(:, 2:size(temp, 2))'; %First column is class labels
trainlabels = double(temp(:, 1))';