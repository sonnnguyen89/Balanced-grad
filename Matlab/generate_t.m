function t = generate_t(ic, M)
% Generates desired outputs as 0 for incorrect class, 1 for correct class.
b = 1;
Nv2=size(ic,1);
% t = ones(Nv,M)*(-b);
t = zeros(Nv2,M);
for p=1:Nv2
    t(p,ic(p)) = b;
end
