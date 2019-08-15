function O = activ(net,f)
%f == 0 --> sigmoid
%f == 1 --> relu
if f == 0
    O = logsig(net);
else
    O = max(net,0);
end