function lsm = logsoftmax(loglh);
[N,T]=size(loglh);
den = logsumexp(loglh);
lsm = loglh-repmat(den,N,1);