
close all;

Ttrain = 50;
Ttest = 1000;
N = 3;
D = 10;
F = 4;

r = 1;
s = 2;
fprintf('Synthesing %i-class, %i-dimensional data, with heteroscedastic rank-%i PPCA covariance model.\n',N,D,F);
fprintf('  training data: %i trails per class.\n',Ttrain);
fprintf('  independent test data: %i trails per class.\n',Ttest);

fprintf('\nStrike any key to continue...');
pause;fprintf('\n\n');




R = orth(randn(D)); % random rotation (and possibly reflection).
Mu = zeros(D,N);
CC = cell(1,N);
for i=1:N;
   Mu(i,i) = s; 
   W = randn(D,F);   
   C = eye(D)+r*D*W*W'/trace(W*W');
   CC{i} = R*(C*D/trace(C))*R';
end;
Mu = R*Mu;

[train,classf_train] = gauss_data(CC,Mu,Ttrain);
[test,classf_test] = gauss_data(CC,Mu,Ttest);

qppca_scores = cell(1,D);
lppca_scores = cell(1,D);
qfa_scores = cell(1,D);
hlda_scores = cell(1,D);
rank = cell(1,D);
dim = cell(1,D);
for i=0:D-1;
   fprintf('\nTraining quadratic PPCA back-end, using %i factors:\n',i);
   [CC,Mu] = train_quadratic_backend(train,classf_train,{'ppca',i});
   fprintf('Done.\n');
   qppca_scores{i+1} = apply_quadratic_backend(test,CC,Mu);;
   
   fprintf('\nTraining linear PPCA back-end, using %i factors:\n',i);
   [Trans,offset] = train_linear_backend(train,classf_train,{'ppca',i});
   fprintf('Done.\n');
   lppca_scores{i+1} = apply_linear_backend(test,Trans,offset);
   
   fprintf('\nTraining quadratic FA back-end, using %i factors:\n',i);
   [CC,Mu] = train_quadratic_backend(train,classf_train,{'fa',i});
   fprintf('Done.\n');
   qfa_scores{i+1} = apply_quadratic_backend(test,CC,Mu);
   
   fprintf('\nTraining HLDA back-end, with output dimension %i:\n',i+1);
   [Trans,CC,Mu] = train_hlda_backend(train,classf_train,i+1);
   fprintf('Done.\n');
   hlda_scores{i+1} = apply_hlda_backend(test,Trans,CC,Mu);
   
   rank{i+1} = int2str(i);
   dim{i+1} = int2str(i+1);
end;


figure(1);
subplot(221);
fprintf('\nEvaluating %i linear PPCA backends:\n',D);
calref_plot(lppca_scores,classf_test,rank);
title('Linear PPCA backend');
xlabel('ppca rank');
fprintf('\nDone, see top left figure 1.\n');

subplot(222);
fprintf('\nEvaluating %i quadratic PPCA backends:\n',D);
calref_plot(qppca_scores,classf_test,rank);
title('Quadratic PPCA backend');
xlabel('fa rank');
fprintf('\nDone, see top right figure 1.\n');

subplot(223);
fprintf('\nEvaluating %i quadratic FA backends:\n',D);
calref_plot(qfa_scores,classf_test,rank);
title('Quadratic FA backend');
xlabel('fa rank');
fprintf('\nDone, see bottom left figure 1.\n');

subplot(224);
fprintf('\nEvaluating %i HLDA backends:\n',D);
calref_plot(hlda_scores,classf_test,dim);
title('HLDA backend');
xlabel('hlda dimension');
fprintf('\nDone, see bottom right, figure 1.\n');
fprintf('(Expand figure for better visibility and drag legends if they obscure bar-graphs.)\n');

fprintf('\nThe Quadratic PPCA backend, which matches the synthetic data model,\n');
fprintf('probably performed best, but since this is random data, anything may happen.\n');
fprintf('You may re-run this example to test it on different random data.\n');

fprintf('\n\nThis is synthetic data, used here just to demonstrate how these tools work.\n');
fprintf('Don''t base conclusions about relative merit of these backends on this data.\n'); 
fprintf('Instead, use the tools demonstrated here to draw conclusions\n'); 
fprintf('from your own real data.\n');



