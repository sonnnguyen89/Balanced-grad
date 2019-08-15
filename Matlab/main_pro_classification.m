%%Select data file and number of iteration
file_id = 1;
Nit = 100;
f_active = 1;
% f_active =1 if for ReLu(), f_active = 0 if for sigmoid()
file_name = 'data/cov.mat';
Nh = 20;
load(file_name);
[E,Et_report,mul,both_side,idx]  = balance_grad_class(x,t,x,t,x,t,Nh,Nit,f_active);
[Ei,Et,ratio] = BP2_nobypass_class(x,t,x,t,x,t,Nh,Nit,f_active);