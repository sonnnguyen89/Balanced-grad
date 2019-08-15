function O = der_O(net,f)
%f == 0 --> sigmoid
%f == 1 --> relu
if f == 0
    O = net.*(1-net);
else
    O = net > 0;
end