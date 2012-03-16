
% closes all graphics windows
close all;

fprintf('\n------------- Multi-class FoCal Demonstration --------------- \n');
fprintf('This script demonstrates the capabilities of the FoCal toolkit \n');
fprintf('of evaluation, fusion and calibration of multi-class recognizers. \n');
fprintf('FoCal calibrates multi-class recognition scores in application-independent\n'); 
fprintf('log-likelihood form, to be suitable for a wide range of identification or detection\n'); 
fprintf('applications.\n\n');

fprintf('This script:\n');
fprintf('- generates synthetic data of varying calibration and refinenemt\n');
fprintf('- fuses three ''systems''\n');
fprintf('- evaluates and graphically compares all systems in terms of calibration and refinenemt.\n\n');


fprintf('Generating training data for 4 different systems\n');

N = 3;    % number of classes
Ttrain = 50; % number of trials per class used for supervised training
Ttest = 1000; % number of independent trials per class used for testing

fprintf('  -System0 (constant likelihoods):\n'); 
fprintf('     - worst possible refinement (no recognition capability),\n'); 
fprintf('     - poor calibration.\n');
const_ll = repmat([1:N]',1,N*Ttest);

fprintf('  -System1:\n'); 
fprintf('     - poor refinement,\n'); 
fprintf('     - calibration: log-likelihoods deflated by factor of 2\n');
[train1c,classf_train] = default_synthesis(N,Ttrain,1); % calibrated likelihoods from synthetic gaussian data
train1 = perturb(2,[0;0;0],train1c);

fprintf('  -System2: \n');
fprintf('     - better refinement, \n');
fprintf('     - calibration: log-likelihoods inflated by factor of 5\n');
[train2c,classf_train] = default_synthesis(N,Ttrain,2); % calibrated likelihoods from synthetic gaussian data
train2 = perturb(1/5,[0;0;0],train2c);

fprintf('  -System3:\n');
fprintf('     - poor refinement, \n');
fprintf('     - calibration: log-likelihoods shifted by [0;2;-2]\n');
[train3c,classf_train] = default_synthesis(N,Ttrain,1); % calibrated likelihoods from synthetic gaussian data
train3 = perturb(1,[0;-2;2],train3c);

fprintf('  -System1+2+3 (fusion of 1,2,3): \n');
fprintf('     - good refinement, \n');
fprintf('     - good calibration\n');

fprintf('\nStrike any key to plot these log-likelihoods...');
pause;fprintf('\n');


as = 10;
figure(1);
subplot(231);threeclass_scatterplot(train1c,classf_train);title('Sys1: calibrated');axis([-as,as,-as,as]);axis('square');grid;
subplot(232);threeclass_scatterplot(train2c,classf_train);title('Sys2: calibrated');axis([-as,as,-as,as]);axis('square');grid;
subplot(233);threeclass_scatterplot(train3c,classf_train);title('Sys3: calibrated');axis([-as,as,-as,as]);axis('square');grid;
subplot(234);threeclass_scatterplot(train1,classf_train);title('Sys1: deflated');axis([-as,as,-as,as]);axis('square');grid;
subplot(235);threeclass_scatterplot(train2,classf_train);title('Sys2: inflated');axis(3*[-as,as,-as,as]);axis('square');grid;
subplot(236);threeclass_scatterplot(train3,classf_train);title('Sys3: shifted');axis([-as,as,-as,as]);axis('square');grid;

fprintf('\nSee Figure 1:\n'); 
fprintf('  Top row of figures show calibrated log-likelihoods:\n');
fprintf('  Bottom row shows perturbed (scaled or shifted) log-likelihoods.\n');
fprintf('    x-axis = log P(data|class_1) - log P(data|class_3)\n');
fprintf('    y-axis = log P(data|class_2) - log P(data|class_3)\n');

fprintf('\nStrike any key to start fusion training...');
pause;fprintf('\n');



[alpha,beta] = train_nary_llr_fusion({train1,train2,train3},classf_train);
fprintf('\nFusion trained:\n');
fprintf(' - combination weights = ');disp(alpha');
fprintf(' - offset vector = ');disp(beta');
fprintf('Notice that combination weights are close to [2,1/5,1], which correct respectively for:\n');
fprintf('  - the log-likelihood deflation of 2.0 of system 1, and \n');
fprintf('  - the log-likelihood inflation by 5.0 of system 2.\n');
fprintf('The offset is close to [0,-2,2], which corrects for: \n');
fprintf('  - the log-likelihood offset of [0,2,-2] of system 3.\n');
fprintf('(For this naive synthetic example, systems are independent, so straight addition of\n');
fprintf('all three calibrated systems gives a calibrated fused score. With real data\n');
fprintf('the weights would also compensate for dependencies between systems.)\n\n');

fprintf('Strike any key to start evaluation of individual and fused systems...');
pause;fprintf('\n');

fprintf('\nGenerating new test data for systems 1,2 and 3.\n');
[test1,classf_test]=default_synthesis(N,Ttest,1);
test1 = perturb(2,[0;0;0],test1);

[test2,classf_test]=default_synthesis(N,Ttest,2);
test2 = perturb(1/5,[0;0;0],test2);

[test3,classf_test]=default_synthesis(N,Ttest,1);
test3 = perturb(1,[0;-2;2],test3);

fprintf('Fusing test data for systems 1,2 and 3.\n');
discr_fusion = apply_nary_lin_fusion({test1,test2,test3},alpha,beta);

figure(2);
fprintf('Evaluating test data for 5 systems: \n\n');
quiet = 1;
cdet_cllr_plot({const_ll,test1,test2,test3,discr_fusion},classf_test,{'0','1','2','3','1+2+3'});

fprintf('\nEvaluation done, see Figure 2.\n');
fprintf('\nObservations:\n');
fprintf('- System 0 is useless: It has some calibration loss, which can be seen with Cllr,\n');
fprintf('    but not with Cdet. Even after calibration has been adjusted,\n');
fprintf('    it does no better than guessing (y-axis = 1).\n');
fprintf('- System 1 has a deflation problem (again visible only with Cllr).\n');
fprintf('- System 2 has an inflation problem (visible with Cdet and Cllr).\n');
fprintf('- System 3 has a shift problem (visible with Cdet and Cllr).\n');
fprintf('- The fusion is much better than the three inputs and has very\n');
fprintf('    little calibration loss.\n');
fprintf('- Cdet and both flavours of Cllr behave qualitatively in much the same way,\n');
fprintf('    except that calibration problems are not always visible with Cdet.\n');

fprintf('\nFor more information about the 6 evaluation meausures displayed in the graphs,\n');
fprintf('  type: help cdet_cllr_plot\n');

