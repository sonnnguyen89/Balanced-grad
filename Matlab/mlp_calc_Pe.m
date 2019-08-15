function Pe = mlp_calc_Pe(ic, y)
% Computes the MSE from t and y. Also returns the error at each output as
% Ei
[C,ic1] = max(y, [], 2);
Pe = sum(ic ~= ic1)*100/size(ic1,1);
