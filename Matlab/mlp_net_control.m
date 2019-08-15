function Wi = mlp_net_control(Xa, Wi, debugging)
% Performs net control on the MLP. Wi is Nhx(N+1).
% Returns the updated Wi matrix.

desired_mean = 0.5;
desired_std_dev = 1;
if(nargin < 3)
    debugging = 0;
end

[Nh,N1] = size(Wi);

net = mlp_calc_net(Xa, Wi);

mean_net = mean(net);
std_net = std(net, 1);

Wi = Wi * desired_std_dev ./ repmat(std_net', [1 N1]);
Wi(:,1) = Wi(:,1) + desired_mean - mean_net' .* desired_std_dev ./ std_net';

if(debugging)
    net = mlp_calc_net(Xa, Wi);
    mean_net = mean(net);
    disp(mean_net);
    std_net = std(net, 1);
    disp(std_net);
end
