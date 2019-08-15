function [z,alpha,f] = find_max(dydz1,dydz2,dy1)
dec = -1*ones(4,1);
alpha = 1;
%old code, brute force, make comment on 08/29/2018
% a_range =(0.01:0.01:2);
% b = (a_range.*T1+1./a_range.*T2)./(a_range.*a_range.*T3 + 1./(a_range.*a_range).*T4+T5);
% [z,a] = max(b);
% a=a/100; %find the real root value that maximize product_z
F1 =sum(sum(dy1.*dydz1));
F2 =sum(sum(dy1.*dydz2));
T1 = F1^2;
T2 = 2*F1*F2;
T3 = F2^2;
T4 = sumsqr(dydz1);
T5 = 2*sum(sum(dydz1.*dydz2));
T6 = sumsqr(dydz2);
   
rt = roots([T1*T5 - T2*T4, 2*(T1*T6 - T3*T4),T2*T6 - T3*T5]);
dec(1) = T1/T4;
dec(2) = T3/T6;
if rt(1) > 0
    dec(3) = (sum(sum((rt(1)*dydz1+dydz2) .* dy1)))^2/sumsqr(rt(1)*dydz1+dydz2);
end
if rt(2) > 0
    dec(4) = (sum(sum((rt(2)*dydz1+dydz2) .* dy1)))^2/sumsqr(rt(2)*dydz1+dydz2);
end
    
[val,idx]=max(dec);
if val > 0
    if idx == 1
        f = 1;
        z = F1/T4;
    elseif idx == 2
        f = 2;
        z = F2/T6;
    else
        f = 3;
        alpha = sqrt(rt(idx-2));
        z = (alpha*F1+F2/alpha)/(alpha^2*T4 + T5 + T6/(alpha^2));
    end
else
    error('shit happen');
end


end