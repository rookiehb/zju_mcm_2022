
Price=csvread('CAO.csv');
Property=csvread('./Decision/PropertyCGB_Poly3.csv');

len=size(Property,1);
DeltaGoldPrice=zeros(1, len-1);
DeltaBitcoin=zeros(1, len-1 );
DeltaAssets=zeros(1, len-1 );
for i=1:len-1
    DeltaGoldPrice(1, i) =  ( Price(i+1+150, 1)-Price(i+150, 1) ) / Price(i+150, 1);
    DeltaBitcoin(1, i) = ( Price(i+1+150, 2)-Price(i+150, 2) ) / Price(i+150, 2);
    DeltaAssets(1, i) =  ( Property(i+1, 4)-Property(i, 4) ) / Property(i, 4);
end
x=1:len-1;
plot(x,DeltaGoldPrice(1,:),'k-o');
hold on;
plot(x ,DeltaBitcoin(1,:),'r-*', 'linewidth',1);
plot(x, DeltaAssets(1,:),'b-+', 'linewidth',1);
legend('Gold','Bitcoin','Assets', 'linewidth',1);
set(gca,'FontSize',20);  %改变图中坐标的大小 20表示坐标显示的大小
grid minor;
box off;

set(gca,'color','non');
xlabel('Day','fontsize',20,'fontweight','bold');
ylabel(' Rate of price change','fontsize',20,'fontweight','bold');
% legend('Total Assets','Cash','Gold','Bitcoin','fontsize',20,'location','northwest','box','on');
xlim([1050,1150]);
ylim([-0.4,0.2]);
% xlim([1200, 1300]);
% ylim([0,15000]);
title('Rate of price change in each day','fontsize',25,'fontweight','bold');
