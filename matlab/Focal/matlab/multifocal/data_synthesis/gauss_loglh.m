function [loglh,classf] = gauss_loglh(C,Mu,trialsPerCLass); 

[data,classf] = gauss_data(C,Mu,trialsPerCLass); 
loglh = gauss_data2loglh(C,Mu,data);