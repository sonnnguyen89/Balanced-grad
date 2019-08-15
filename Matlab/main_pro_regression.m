%%Select data file and number of iteration
file_id = 1;
Nit = 100;

% f_active =1 if for ReLu(), f_active = 0 if for sigmoid()
f_active = 1;  
if file_id ==1
    file_name = 'data/RosenBrock.mat';
    Nh = 12;
elseif file_id == 2
    file_name = 'data/inverse9.mat';
    Nh = 12;
elseif file_id ==3
    file_name = 'data/superconductivity.mat';
    Nh = 20;
elseif file_id ==4
    file_name = 'data/tall_file.mat';
    Nh = 50;
end

load(file_name);
[E,Et_report,mul,both_side,idx]  = balance_grad(x,t,x,t,x,t,Nh,Nit,f_active);
[Ei,Et,ratio] = BP2_nobypass(x,t,x,t,x,t,Nh,Nit,f_active);