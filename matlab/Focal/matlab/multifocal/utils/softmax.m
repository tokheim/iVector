function p = softmax(loglh);
p = exp(logsoftmax(loglh));