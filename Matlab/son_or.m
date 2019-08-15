function t = son_or(y,ic)
%A fast, vectorized version of output-reset
%Created by Son Nguyen, May 22nd 2019
%Nit should be a small integer, such as 3
Nit = 3;
M = size(y,2);
t = generate_t(ic,M);
z = y - t;
ap= (mean(z,2));
for it=1:Nit
    %consider all are incorrect class
    d1 = z-ap;
    d2 = z < ap;
    di = d1.*d2.*(1-t);
    %consider all are conrrect class
    d2 = z > ap;
    dii = d1.*d2.*t;
    d = di + dii;
    
    ap= mean(z-d,2);
end
 t = t + ap + d;
 

