function net = mlp_calc_net(Xa, Wi)
% net = mlp_calc_net(Xa, Wi) Calculates the 
% net function using the inputs and input 
% weights. Wi is Nhx(N+1)

net = Xa * Wi';
