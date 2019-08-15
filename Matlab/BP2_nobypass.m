function [Ei,Et,ratio] = BP2_nobypass(x,t,x_v,t_v,x_t,t_t,Nh,Nit,f)
%(x,t), (x_v,t_v), (x_t,t_t) are training, validation, and testing data
%Nh: number of hidden units
%Nit: number of iterations
% f_active =1 if for ReLu(), f_active = 0 if for sigmoid()
input_means = mean(x);
x = bsxfun(@minus, x, input_means);
x_v = bsxfun(@minus, x_v, input_means);
x_t = bsxfun(@minus, x_t, input_means);

[Nv,N] = size(x);
M = size(t,2);
Nv_v = size(x_v,1);
Nv_t = size(x_t,1);
Xa=[ones(Nv,1) x];
Xa_v = [ones(Nv_v,1) x_v];
Xa_t = [ones(Nv_t,1) x_t];
mlp_randn(0,0,1);
W=mlp_randn(Nh,N+1);
W = mlp_net_control(Xa, W, 0);
O=[ones(Nv,1) activ(Xa*W',f)];

R = (O' * O) / Nv;
C = (O' * t) / Nv;
Wo=OLS2(R,C);
%Wo = mlp_randn(M,Nh+1);

y=O*Wo';
dy=2*(t-y);
RR = Xa' * Xa / Nv;
dO =  der_O(O,f);
Ei=zeros(1,Nit);
ratio=zeros(1,Nit);
Ei(1,1)=(1/Nv)*sumsqr(t-y);
for it=1:Nit
    g = (1/Nv)*(dO(:,2:end).*(dy*Wo(:,2:end)))' * Xa;
    goh = (1/Nv)*dy' * O;
    OO = (1/Nv)*(O'*O);
    g=OLS2(RR,g');
    goh = OLS2(OO,goh');
    
    dydz1 = [ones(Nv,1) (dO(:,2:end).*(Xa*g'))]* Wo';
    dydz2 = O * goh';
   
    G=zeros(2,1);
    H=zeros(2,2);
    G(1,1)=(sum(sum(dy.*dydz1)))/Nv;
    G(2,1)= (sum(sum(dy.*dydz2)))/Nv;
    H(1,1)=(sumsqr(dydz1))*2/Nv;
    H(1,2)=(sum(sum(dydz1.*dydz2)))*2/Nv;
    H(2,1)=H(1,2);
    H(2,2)=(sumsqr(dydz2))*2/Nv;
    Z=OLS2(H,G);
    Z = abs(Z);
    Wo=Wo+Z(2)*goh;
    W=W+Z(1)*g;
    
    O=[ones(Nv,1) activ(Xa*W',f)];
    y=O*Wo';
    
    dy=2*(t-y);
    dO =  der_O(O,f);
    Ei(1,it+1)=(1/Nv)*sumsqr(t-y);
    ratio(it) = det(H);
    %fprintf('%f \t\t %f\t\t %f\n',Z(1),Z(2),det(H));
    if Ei(1,it+1) > Ei(1,it)
        fprintf('%d\t\t%f\t\t%f\n',it, Ei(it),Ei(it+1));
    end
end
Et = (1/Nv_t)*sumsqr([ones(Nv_t,1) activ(Xa_t*W',f)]*Wo' - t_t);