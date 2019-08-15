function [E,Et_report,mul,both_side,idx]  = balance_grad_class(x,ic,x_v,ic_v,x_t,ic_t,Nh,Nit,f_active)
tic
input_means = mean(x);
x = bsxfun(@minus, x, input_means);
x_v = bsxfun(@minus, x_v, input_means);
x_t = bsxfun(@minus, x_t, input_means);
t = generate_t(ic,max(ic));
[Nv,N] = size(x);
M = size(t,2);
Nv_v = size(x_v,1);
Nv_t = size(x_t,1);
Xa=[ones(Nv,1) x];
Xa_v = [ones(Nv_v,1) x_v];
Xa_t = [ones(Nv_t,1) x_t];
mlp_randn(0,0,1);
W = mlp_randn(Nh,N+1);
W = mlp_net_control(Xa, W, 0);
O=[ones(Nv,1) activ(Xa*W',f_active)];

R = (O' * O) / Nv;
C = (O' * t) / Nv;
Wo=OLS2(R,C);

y=O*Wo';
t = son_or(y, ic);
dy=2*(t-y);
dO =  der_O(O,f_active);
E=zeros(1,Nit);
Et=E;
Ev = E;

E(1,1)=(1/Nv)*sumsqr(t-y);
both_side = 0;
RR = Xa' * Xa / Nv;
for  it=1:Nit
    %Calculate 2 gradient matrix
    g = (1/Nv)*(dO(:,2:end).*(dy*Wo(:,2:end)))' * Xa;
    %Nv*M*Nh + Nv*Nh + Nv*Nh*(N+1) + Nh*(N+1)
    goh = (1/Nv)*dy' * O;
    %Nv*M*Nh+Nh*M
    OO = (1/Nv)*(O'*O);
    %Nv*(Nh+1)^2
    g=OLS2(RR,g');
    %(N+1)*(N+2)*(Nh+(1/6)*(N+1)*(2*N+3)+3/2)
    goh = OLS2(OO,goh');
    %(Nh+1)*(Nh+2)*(M+(1/6)*(Nh+1)*(2*Nh+3)+3/2)
    
    dydz1 = [ones(Nv,1) (dO(:,2:end).*(Xa*g'))]* Wo';
    dydz2 = O * goh';
    dy1 = dy/2;
    %Nv*(N+1)*Nh + Nv*Nh+ Nv*(Nh+1)*M + Nv*Nh*M
    
    [z,a,f] = find_max(dydz1,dydz2,dy1);
    %7*Nv*M
    %     dydz = dydz1 + dydz2;
    %     J = sum(sum(dydz.* dy));
    %     H = sumsqr(dydz);
    %     z = J/H;
    %     a=1;
    if f == 1
        W = W + z*g*a;
    elseif f == 2
        Wo = Wo + z*goh/a;
    else 
        W = W + z*g*a;
        Wo = Wo + z*goh/a;
        both_side = both_side + 1;
    end
    % (N+1)*Nh + (Nh+1)*M
    
    O=[ones(Nv,1) activ(Xa*W',f_active)];
    y=O*Wo';
    t = son_or(y, ic);
    
    dy=2*(t-y);
    dO =  der_O(O,f_active);
    E(1,it+1)=(1/Nv)*sumsqr(t-y);
    
    O_v=[ones(Nv_v,1) activ(Xa_v*W',f_active)];
    y_v = O_v*Wo';
    Ev(it) = mlp_calc_Pe(ic_v, y_v);
    
    O_t=[ones(Nv_t,1) activ(Xa_t*W',f_active)];
    y_t = O_t*Wo';
    Et(it) = mlp_calc_Pe(ic_t, y_t);
    
end %%%%%%%%%%%%%%%%%%%%%%%%%
Ev_smooth = (1/3)*(Ev+[Ev(1) Ev(1:end-1)]+[Ev(2:end) Ev(end)]);
[~,idx]=min(Ev_smooth);
Et_report = Et(idx);

mul_s = Nv*M*Nh + Nv*Nh + Nv*Nh*(N+1) + Nh*(N+1) + Nv*M*Nh+Nh*M + Nv*(Nh+1)^2 +(N+1)*(N+2)*(Nh+(1/6)*(N+1)*(2*N+3)+3/2) + (Nh+1)*(Nh+2)*(M+(1/6)*(Nh+1)*(2*Nh+3)+3/2) + Nv*(N+1)*Nh + Nv*Nh+ Nv*(Nh+1)*M + Nv*Nh*M + 7*Nv*M + (N+1)*Nh + (Nh+1)*M;
mul = (1:Nit)*mul_s;
both_side = both_side/Nit;
toc
% fprintf('both side = %d\n', both_side);