%%
clc;
close;
clear;
Gold=xlsread('LBMA-GOLD(1)(1)(1).csv');
Bitcoin=xlsread('BCHAIN-MKPRU.csv');
len=size(Gold,1);
previousDay=7;
startDate=16;
minError=9999;

predictGold=zeros(len,10);
for day=startDate+1:1:len-1
    minError=9999;res=0;
    for i=1:5
        x=day-previousDay:1:day;
        y=Gold(x);
        fun=polyfit(x,y,i);%得到i从1到5的多项式
        if i==1
            bestfun=fun;
        end
        Y=polyval(fun,x);%计算拟合函数在x处的值。
        if sum((Y-y).^2)<minError%利用理论值与实际值做平方差和，如果值小与0.1即满足需求
            res=i;
            minError=sum((Y-y).^2);
            bestfun=fun;
        end
    end
    for j=1:10
        predictGold(day,j)=polyval(bestfun,day+j);
    end
end
csvwrite('polyGoldFindbestTp.csv', predictGold);


predictBitcoin=zeros(len,10);
for day=startDate+1:1:len-1
    minError=9999;res=0;
    for i=1:5
        x=day-previousDay:1:day;
        y=Bitcoin(x);
        fun=polyfit(x,y,i); %得到i从1到5的多项式
        if i==1
            bestfun=fun;
        end
        Y=polyval(fun,x);%计算拟合函数在x处的值。
        if sum((Y-y).^2)<minError%利用理论值与实际值做平方差和，如果值小与0.1即满足需求
            res=i;
            minError=sum((Y-y).^2);
            bestfun=fun;
        end
    end
    for j=1:10
        predictBitcoin(day,j)=polyval(bestfun,day+j);
    end
end
csvwrite('polyBitcoinFindbestTp.csv', predictBitcoin);


