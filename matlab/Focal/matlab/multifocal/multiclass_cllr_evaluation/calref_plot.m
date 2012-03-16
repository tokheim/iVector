function calref_plot(scores,classf,names,quiet);
% CALREF_PLOT: Bar-graph comparison, in terms of calibration 
%              and refinement, between multiple recognizers.
%   Usage: 
%         CALREF_PLOT(scores, classf, names);
%         CALREF_PLOT({scores1, scores2, ...}, classf, {'sys1','sys2', ...});
%         CALREF_PLOT({scores1, scores2, ...}, classf, {'sys1','sys2', ...},quiet);
%  
%   Input parameters:
%      scores  : 1 by K cell array, where each element is an N by T matrix. K >= 1
%                These are N scores for each of T trials for each of K input systems: 
%                  scores{k}(i,t) is the score of system k, for recognizing class i in trial t.
%                Scores should be calibrated to have relative log-likelihood nature: 
%                  scores{k,i,t} = log P(trial_t | class_i, system_k) - offset_tk
%
%      classf  : 1 by T row, with elements in {1,2, ..., N}.
%                These are the true class labels for the trials.
%
%      quiet   : 0 show optimization iterations
%                1 don't show optimization
%                optional: default  1

if nargin<4
   quiet = 1;
end;


K = length(scores);
if length(names) ~= K
   error('illegal parameter: length of ''scores'' and ''names'' should be equal');
end;

[N,T] = size(scores{1});

raw = zeros(1,K);
opt = zeros(1,K);

syslen = length('default');
for k=1:K
   if length(names{k})>syslen, syslen = length(names{k}); end;
end;
width = 17;

if quiet
   fprintf('%-*s : %*s + %*s = %*s\n',syslen,'System',width,'Refinement loss',width,'Calibration loss',width,'Total loss');
   for i=1:syslen+3*width+10,fprintf('-');end;fprintf('\n');
end;   
if quiet
   fprintf('%-*s : %*.5f + %*.5f = %*.5f \n',syslen,'default',width,log(N)/log(2),width,0,width,log(N)/log(2));
end;   
for k=1:K
   [opt(k),raw(k)] = multiclass_min_cllr(scores{k},classf,quiet);   
   if quiet
      fprintf('%-*s : %*.5f + %*.5f = %*.5f \n',syslen,names{k},width,opt(k),width,raw(k)-opt(k),width,raw(k));
   end;   
end;
if quiet
   for i=1:syslen+3*width+10,fprintf('-');end;fprintf('\n');
end;

if ~quiet
   fprintf('\n');
   fprintf('%-*s : %*s + %*s = %*s\n',syslen,'System',width,'Refinement loss',width,'Calibration loss',width,'Total loss');
   for i=1:syslen+3*width+10,fprintf('-');end;fprintf('\n');
   fprintf('%-*s : %*.5f + %*.5f = %*.5f \n',syslen,'default',width,log(N)/log(2),width,0,width,log(N)/log(2));
   for k=1:K
      fprintf('%-*s : %*.5f + %*.5f = %*.5f \n',syslen,names{k},width,opt(k),width,raw(k)-opt(k),width,raw(k));
   end;   
   for i=1:syslen+3*width+10,fprintf('-');end;fprintf('\n');
end;


raw=[0,raw];
opt=[0,opt];
def = [log(N)/log(2),zeros(1,K)];
K = K+1;
%names = {'logN/log2',names{:}};
names = {'-',names{:}};

colormap([ 0.4,0.4,0.4; 0,0.7,0; 1,0,0 ]);
bar([def;opt;raw-opt]','stacked');
title('Calibration and Refinement');
xlabel('system');
ylabel('C_{llr} [bits]');
set(gca,'xtick',[1:K]);
set(gca,'xticklabel',names);
grid;
%legend('default loss','refinement loss','calibration loss',3);
legend('logN/log2','refinement loss','calibration loss');
