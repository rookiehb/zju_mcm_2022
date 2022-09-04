%%
% 灰色关联度矩阵模型
clc;
close;
clear all;
% 控制输出结果精度
format short;

Gold=xlsread('LBMA-GOLD(1)(1)(1).csv');
Bitcoin=xlsread('BCHAIN-MKPRU.csv');
len=size(Gold,1);

% 数据标准化处理
GoldData = zeros(len,1);
BitcoinData = zeros(len,1);

for i = 1:len
    GoldData(i,1) = Gold(i,1)/sum(Gold);
    BitcoinData(i,1) = Bitcoin(i,1)/sum(Bitcoin);
end

% 保存中间变量，亦可省略此步，将原始数据赋予变量data
% deltaX=GoldData-BitcoinData;
% deltaX=abs(deltaX);

% bestk=0;
% beststep=0;
% maxR=0;
% x表示时间； y表示最优的b； z表示最好的关联度

LookBacks=150;preparation=501;bestR=0;beststep=0;
x=preparation:len; y=[];z=[];

for step=preparation:len
    bestR=0;beststep=0;
    for tempt=step-500:step-LookBacks
        
        deltaX=zeros(LookBacks,1);
        % 固定比特币  调节黄金
        
        
        deltaX=BitcoinData(tempt:tempt+LookBacks,1)-GoldData(step-LookBacks:step,1);
        deltaX=abs(deltaX);

        min_min=min(min(deltaX));
        max_max=max(max(deltaX));
        resolution=0.5;

        coefficient=0;
        for i=1:LookBacks
            coefficient=coefficient+(min_min+resolution*max_max)./(deltaX(i,1)+resolution*max_max);
        end
        r=coefficient/LookBacks;
        
        if r>bestR
            beststep=tempt;
            bestR=r;
        end
    end
    y=[y beststep];
    z=[z bestR];
end
plot3(x,y,z)
colormap('hot')
grid on
box on

out=[x;  y];
out = out';
csvwrite('Goldfix.csv',out);



