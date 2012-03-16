function loglh = gauss_data2loglh(C,Mu,data); 

if iscell(C) % heteroscedastic
  loglh = apply_quadratic_backend(data,CC,Mu);
else % homoscedastic
  [Trans,offset] = compose_linear_backend(C,Mu);
  loglh = apply_linear_backend(data,Trans,offset);
end;
