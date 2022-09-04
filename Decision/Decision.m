%%
clear;clc;close all;
global NP Iteration gen Pc Pm Gap X  Z V Parameters PropertyCGB Strategy;
Parameters=csvread('T.csv');

% 现在拥有的财产
PropertyCGB=zeros(1673+1, 4);
PropertyCGB(1,1)=1000;PropertyCGB(1,2)=0;PropertyCGB(1,3)=0;PropertyCGB(1,4)=1000;

Strategy=zeros(1673+1,  2); % 第一列是TradeGold，第二列是TradeBitcoin
% 代表的第一天
Strategy(1,1)=0;Strategy(1,2)=0;

DAY=150; %开始的天数
NP=5000; %种群中个体数量
Iteration=50; %总迭代次数
gen=1;%当前迭代次数
Pc=0.75;%交叉概率
Pm=0.05;%变异概率
Gap = 0.9;%每次丢弃父代百分之10，以便加入百分之10的子代
X=zeros(NP,2);


for day = 2:1673
    
    for i = 1:NP    %初始化种群
        Dna=encode(day);   %随机生成种群坐标
        X(i,:)=Dna;
    end
    
    while (gen<=Iteration) %迭代过程
        Z = Fitness(NP,day); % 计算适应度
        parentselector = Select(Z,NP,X,Gap);% 轮盘赌算法 产生90个下一代
        kidselector = cross (parentselector,NP,Pc,Gap); % 通过父辈的交叉随机算出10个子辈
        X = [parentselector;kidselector];% 子辈父辈重新整合在一起
        V = Variation(NP,Pm,X);% 下一代的随机变异
        X = V;
        gen=gen+1;
    end
    
    bestFitness=max(max(Z));
    [row, col] = find(Z==bestFitness);
    Strategy(day, :) = X(row(1,1), : );
%     PropertyCGB(day, 1) = PropertyCGB(day-1,1)-sign( Strategy(day, 1) )*Strategy(day, 1)-sign( Strategy(day, 2) )*Strategy(day, 2 );
    if  Strategy(day,1)==0 && Strategy(day,2)==0
        PropertyCGB(day, 1) = PropertyCGB(day-1,1);
        PropertyCGB(day, 2) = PropertyCGB(day-1, 2);
        PropertyCGB(day, 3) = PropertyCGB(day-1, 3);
    else
        % PropertyCGB(day, 1) = PropertyCGB(day-1,1)-Strategy(day,1)-Strategy(day,2)-abs( Strategy(day,1))*0.0001-abs(Strategy(day,2))*0.0002;
        PropertyCGB(day, 1) = PropertyCGB(day-1,1)-Strategy(day,1)-Strategy(day,2)-abs(Strategy(day,2))*0.0002-abs(Strategy(day,1))*0.0004;
        PropertyCGB(day, 2) = PropertyCGB(day-1, 2) +  Strategy(day, 1);
        PropertyCGB(day, 3) = PropertyCGB(day-1, 3) +  Strategy(day, 2);
    end
     % PropertyCGB(day,4)=PropertyCGB(day,1) + PropertyCGB(day,2)*Parameters(day,1)/Parameters(day-1,1) + PropertyCGB(day,3)*Parameters(day,2)/Parameters(day-1,2);
     PropertyCGB(day,2) = PropertyCGB(day,2)*Parameters(day,1)/Parameters(day-1,1);
     PropertyCGB(day,3) = PropertyCGB(day,3)*Parameters(day,2)/Parameters(day-1,2);
     PropertyCGB(day,4)=PropertyCGB(day,1) + PropertyCGB(day,2) + PropertyCGB(day,3);

end

% csvwrite('./SensitiveAnalysis/Strategy_w0.3_6.csv', Strategy);
csvwrite('./SensitiveAnalysis/PropertyCGB_alphaGold4_2.csv', PropertyCGB);

%%  计算利润
function Z = Fitness(NP,day)
global Parameters PropertyCGB X;
Z=zeros(NP,1);
% 先写一个面积的预测
for i=1:NP
% profit = PropertyCGB(day-1, 1)+ PropertyCGB(day-1, 2) + PropertyCGB(day-1, 3)  + ( X(i,1)+PropertyCGB(day-1,2) )*Parameters(day-1, 8) + ( X(i, 2)+PropertyCGB(day-1,3) )*Parameters(day-1, 9)-abs(X(i,1))*0.01-abs(X(i,2))*0.02;
    
    profit1 = PropertyCGB(day-1, 1)-X(i,1)-X(i,2) + (PropertyCGB(day-1, 2)+X(i, 1))*(Parameters(day-1, 3)+1) +(PropertyCGB(day-1, 2) + X(i, 2))*(Parameters(day-1, 4)+1)-abs(X(i,2))*0.0002-abs(X(i,1))*0.0004;
    profit2 = PropertyCGB(day-1, 1)+PropertyCGB(day-1, 2)*(Parameters(day-1, 3)+1)+PropertyCGB(day-1, 3)*(Parameters(day-1, 4)+1);
    
    if  PropertyCGB(day-1, 1)<0 || PropertyCGB(day-1, 2)<0 || PropertyCGB(day-1, 3)<0
        profit1=profit1*0.1;
    end
    
    if  X(i, 1) + PropertyCGB(day-1, 2) < 0 || X(i, 2) + PropertyCGB(day-1, 3)<0 || PropertyCGB(day-1,1)-X(i,1)-X(i,2)<0
        profit1 = profit1*0.1;
    end
    
    w=0.2;
    if  ( PropertyCGB(day-1, 1)-X(i,1)-X(i,2)  )  /  PropertyCGB(day-1, 4)  < w
        profit1=profit1*0.1;
    end
    
    Cov = X(i,1)*Parameters(day-1, 3)+ X(i,2)*Parameters(day-1, 4)-sqrt( Parameters(day-1, 5)*X(i,1).^2+Parameters(day-1, 6)*X(i,2).^2+2*Parameters(day-1, 7)*X(i,2)*X(i,1) );
    p=-0.10;
    if Cov/( PropertyCGB(day-1,1)+PropertyCGB(day-1,2)+PropertyCGB(day-1,3) )<p
        profit1=profit1*0.1;
    end
    
    q=-0.04;
    Risk = X(i,1)* ( Parameters(day-1,3)-  Parameters(day-1,8) ) + X(i,2)*( Parameters(day-1,4)-  Parameters(day-1,9) );
     if  Risk/PropertyCGB(day-1, 4) < q
        profit1=profit1*0.1;
     end
     
     if profit1>profit2
        profit = profit1;
    else
        profit = profit2;
        X(i,1)=0;X(i,2)=0;
    end
   
    if profit < 0
        profit = profit2;
        X(i,1)=0;X(i,2)=0;
        continue;
    end
    Z(i,1) = profit;
end

end

%%  %轮盘赌选择下一代
function parentselector = Select( Z, NP, X, Gap ) 
parentselector = zeros(NP*Gap,2);
sumfitness = sum (Z);
accP = cumsum(Z/sumfitness); %累积概率
for n = 1:NP*Gap
    matrix = find (accP>rand); %找到比随机数大的概率
    if isempty(matrix)
        continue;
    end
    temp = X(matrix(1),:);
    parentselector(n,:) = temp;
end
end
%% %交叉
function  kidselector = cross (parentselector,NP,Pc,Gap)
kidselector = zeros( int16(NP*(1-Gap)),2); 
n=1;
father = zeros(1,2);
mother = zeros(1,2);
while n<=int16(NP*(1-Gap))
    father(1,:) = parentselector( ceil(rand*NP*Gap), : );  % 随机产生父辈母辈
    mother(1,:) = parentselector( ceil(rand*NP*Gap), : );
    randNum = rand();
    randPos = randi(2);
    if rand < Pc    % 根据概率进行交叉
        father(1, randPos) = mother(1, randPos)*randNum+(1-randNum)*father(1, randPos);
        kidselector(n,:) = father(1,:);
        n=n+1;
    end
end
end
%%
function V = Variation(NP, Pm, X)  %变异
V = zeros(NP,2);
for n=1:NP
    if ( rand < Pm )
        randPos = randi(2);
        X(n, randPos) = (rand()+1)*X(n, randPos); % 变异采用一位的0 1变换
%     elseif rand < Pm*2
%         randPos = randi(2);
%         X(n, randPos) = rand()*X(n, randPos); % 变异采用一位的0 1变换
    end
end
V = X;
end
%%
function Dna=encode(day)  % 初始化种群
global Parameters PropertyCGB;
% range = PropertyCGB(day-1, 1)*0.7 + PropertyCGB(day-1, 2)*0.15 + PropertyCGB(day-1, 3)*0.15;

if rand() < 0.5
    TradeGold = -PropertyCGB(day-1, 2)*rand()*0.7;
else
    TradeGold = PropertyCGB(day-1, 1)*rand()*0.7;
end

if mod(day,7)==5 || mod(day,7)==6
    TradeGold = 0;
end

if rand() < 0.5
    TradeBitcoin = -PropertyCGB(day-1, 3)*rand()*0.7;
else
    TradeBitcoin = PropertyCGB(day-1, 1)*rand()*0.7;
end

% if Parameters(day-1, 8)<0.01 
%     TradeGold = -TradeGold;
% end
% if Parameters(day-1, 9)<0.02
%     TradeBitcoin = -TradeBitcoin;
% end

Dna = [TradeGold TradeBitcoin];

end
