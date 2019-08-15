load('data/oh7.mat');
[Ei,Et,det] = BP2_nobypass(x,t,x,t,x,t,10,100,1);
%[Ei1,Et1,det1] = steepest_descent(x,t,x,t,x,t,10,100,1);
%[Ei1,Et] = balance_grad(x,t,x,t,x,t,10,100,1);

% ax1 = subplot(2,1,1); % top subplot
% plot(ax1,Ei,'b', 'marker','+','MarkerSize',3);
% hold on
% plot(ax1,Ei1,'g', 'marker','p','MarkerSize',3);
% legend('BP2','Balanced gradient');
% title(ax1,'MSE vs iterations');
% ylabel(ax1,'MSE');

% ax2 = subplot(2,1,2); % bottom subplot
% plot(ax2, det,'r', 'marker','d','MarkerSize',3);
% title(ax2,'Hessian''determinant');

figure(2)
plot(Ei(1:end),'b', 'marker','+','MarkerSize',3);
hold on
%plot(Ei1(1:end),'g', 'marker','p','MarkerSize',3);
plot(det/2000,'r', 'marker','d','MarkerSize',3);
%legend('BP2','Balanced gradient','|H|/2000 for BP2');
legend('BP2','|H|/2000 for BP2');
xlabel('iterations');
ylabel('Training MSE');
grid on
%saveas(gcf,'BP2_2','epsc')