clc;
close;
clear all;

Gold=xlsread('LBMA-GOLD(1)(1)(1).csv');
Bitcoin=xlsread('BCHAIN-MKPRU.csv');
len=size(Gold,1);
% 数据标准化处理

LookBacks=30;
preparation=90;

bestR=0;bestStep=0;bestTempt=0;

for step=preparation:len
    for tempt=1:step-LookBacks
        A=Gold(step-LookBacks:step,1);
        B=Bitcoin(tempt:tempt+LookBacks,1);
        [r, p]=corr(A',B');
        if r>bestR
            bestR=r;
            bestStep=step;
            bestTempt=tempt;
        end
    end
end
