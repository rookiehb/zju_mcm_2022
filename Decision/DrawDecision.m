% 大概要画三十个图
 PropertyCGB_T1=csvread('PropertyCGB_T1.csv');
PropertyCGB_T2=csvread('PropertyCGB_T2.csv');
 PropertyCGB_T3=csvread('PropertyCGB_T3.csv');
 PropertyCGB_T4=csvread('PropertyCGB_T4.csv');
 PropertyCGB_T5=csvread('PropertyCGB_T5.csv');
 PropertyCGB_T6=csvread('PropertyCGB_T6.csv');
 PropertyCGB_T7=csvread('PropertyCGB_T7.csv');
 PropertyCGB_T8=csvread('PropertyCGB_T8.csv');
 PropertyCGB_T9=csvread('PropertyCGB_T9.csv');
 PropertyCGB_T10=csvread('PropertyCGB_T10.csv');
 
PropertyCGB_Poly1=csvread('PropertyCGB_Poly1.csv');
PropertyCGB_Poly2=csvread('PropertyCGB_Poly2.csv');
PropertyCGB_Poly3=csvread('PropertyCGB_Poly3.csv');
PropertyCGB_Poly4=csvread('PropertyCGB_Poly4.csv');
PropertyCGB_Poly5=csvread('PropertyCGB_Poly5.csv');
PropertyCGB_Poly6=csvread('PropertyCGB_Poly6.csv');
PropertyCGB_Poly7=csvread('PropertyCGB_Poly7.csv');
PropertyCGB_Poly8=csvread('PropertyCGB_Poly8.csv');
PropertyCGB_Poly9=csvread('PropertyCGB_Poly9.csv');
PropertyCGB_Poly10=csvread('PropertyCGB_Poly10.csv');

PropertyCGB_LSTM1=csvread('PropertyCGB_LSTM1.csv');
PropertyCGB_LSTM2=csvread('PropertyCGB_LSTM2.csv');
PropertyCGB_LSTM3=csvread('PropertyCGB_LSTM3.csv');
PropertyCGB_LSTM4=csvread('PropertyCGB_LSTM4.csv');
PropertyCGB_LSTM5=csvread('PropertyCGB_LSTM5.csv');
PropertyCGB_LSTM6=csvread('PropertyCGB_LSTM6.csv');
PropertyCGB_LSTM7=csvread('PropertyCGB_LSTM7.csv');
PropertyCGB_LSTM8=csvread('PropertyCGB_LSTM8.csv');
PropertyCGB_LSTM9=csvread('PropertyCGB_LSTM9.csv');
PropertyCGB_LSTM10=csvread('PropertyCGB_LSTM10.csv');

 LastDay=size(PropertyCGB_T4, 1);
day = 151:LastDay+150;
y = PropertyCGB_LSTM2( : , 4);
yC = PropertyCGB_LSTM2( : , 1);
yG = PropertyCGB_LSTM2( : , 2);
yB = PropertyCGB_LSTM2( : , 3);


figure('name','Assets on different days','color',[1 1 1]);

subplot(4,1,1);
plot(day ,y ,'b-','linewidth',1);
set(gca,'FontSize',20);  %改变图中坐标的大小 20表示坐标显示的大小
grid minor
box off
set(gca,'color','non');
% xlabel('Day','fontsize',20,'fontweight','bold');
ylabel('Money ','fontsize',20,'fontweight','bold');
% legend('Total Assets','Cash','Gold','Bitcoin','fontsize',20,'location','northwest','box','on');
xlim([151, LastDay+145]);
% xlim([1200, 1300]);
ylim([0,15000]);
title('Total assets on different days','fontsize',25,'fontweight','bold');
% hold on;




subplot(4,1,2);
plot(day ,yC ,'m-','linewidth',1);
set(gca,'FontSize',20);  %改变图中坐标的大小 20表示坐标显示的大小
grid minor
box off
set(gca,'color','non');
% xlabel('Day','fontsize',20,'fontweight','bold');
ylabel('Money ','fontsize',20,'fontweight','bold');
% legend('Total Assets','Cash','Gold','Bitcoin','fontsize',20,'location','northwest','box','on');
xlim([151, LastDay+145]);
% xlim([1200, 1300]);
ylim([0,15000]);
title('Cash on different days','fontsize',25,'fontweight','bold');




subplot(4,1,3);
plot(day ,yG,'y-','linewidth',1);
set(gca,'FontSize',20);  %改变图中坐标的大小 20表示坐标显示的大小
grid minor
box off
set(gca,'color','non');
% xlabel('Day','fontsize',20,'fontweight','bold');
 ylabel('Money ','fontsize',20,'fontweight','bold');
% legend('Total Assets','Cash','Gold','Bitcoin','fontsize',20,'location','northwest','box','on');
xlim([151, LastDay+145]);
% xlim([1200, 1300]);
ylim([0,15000]);
title('Gold on different days','fontsize',25,'fontweight','bold');



subplot(4,1,4);
plot(day ,yB,'r-','linewidth',1);
xlabel('Day','fontsize',20,'fontweight','bold');
ylabel('Money ','fontsize',20,'fontweight','bold');
% legend('Total Assets','Cash','Gold','Bitcoin','fontsize',20,'location','northwest','box','on');
xlim([151, LastDay+145]);
% xlim([1200, 1300]);
ylim([0,15000]);
title('Bitcoin on different days','fontsize',25,'fontweight','bold');
set(gca,'FontSize',20);  %改变图中坐标的大小 20表示坐标显示的大小
grid minor
box off
set(gca,'color','non');


% 
% % set(gca,'XTickLabel',[150:200:LastDay+150]);
% set(gca,'FontSize',20);  %改变图中坐标的大小 20表示坐标显示的大小
% grid minor
% box off
% set(gca,'color','non');

