function [dir,count] = rcg_dir(old_dir, grad, old_grad, count, quiet)
% Compute the new conjugate direction.

if nargin<5
   quiet = 0;
end;

dim = length(grad);

if (dim==1) | (count == dim),
   dir = grad;
   count = 0;
   return;
end;   

delta = grad - old_grad;
den = old_dir'*delta;
if (den==0)
   dir = grad; % restart
   if ~quiet
      fprintf('|delta|=%g\n',sqrt(delta'*delta));   
      fprintf('conjugate gradient restart, because denominator = 0\n');   
   end;
else   
   
  % Hestenes-Stiefel
  beta = (grad'*delta) / den;

  % Polak-Ribiere
  %beta = -grad'*(grad - old_grad) / (old_grad'*old_grad);
  
  % Fletcher-Reeves
  %beta = -(grad'*grad) / (old_grad'*old_grad);
  
  dir = grad - beta*old_dir;
  
  if (dir'*dir==0)
     dir = grad; % restart
     if ~quiet
        fprintf('conjugate gradient restart, because u = 0\n');   
     end;
  end;
  if (dir'*grad <= 0)
     if ~quiet
        fprintf('conjugate gradient restart, because  g''u = %g <= 0\n',dir'*grad);   
     end;   
     dir = grad; % restart
  end;
  
  count = count + 1;
  
end;
