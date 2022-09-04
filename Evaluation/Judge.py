import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import pylab as pl
from datetime import datetime

#Read
filename_B = 'data/BCHAIN-MKPRU.csv'
filename_G = 'data/LBMA-GOLD.csv'

scale = 1
with open(filename_B) as fb:
    reader = csv.reader(fb)
    header_row = next(reader)

    price_bit=[]
    dates = []
    for row in reader:
        single_price_bit = float(row[1])
        price_bit.append(single_price_bit/scale)
        #current_date = datetime.strptime(row[0],'%d/%m/%Y')
        #dates.append(current_date)3
with open(filename_G) as fg:
    reader = csv.reader(fg)
    header_row = next(reader)

    price_gold = []
    for row in reader:
        single_price_gold = float(row[1])
        price_gold.append(single_price_gold)

#Prediction
L = len(price_bit)
days = range(0, L)
predict_B_3d_highst = [0]
predict_G_3d_highst = [0]

for d in range(1, L):
    predict_B_3d_highst.append(price_bit[d-1] + 3 * (price_bit[d]-price_bit[d-1]))
    predict_G_3d_highst.append(price_gold[d-1] + 3 * (price_gold[d]-price_gold[d-1]))
'''
for d in days:
    y = price_bit[0:d+1]
    x = np.linspace(0, d, d+1)
    kind = "cubic"
    # "nearest","zero"为阶梯插值
    # slinear 线性插值
    # "quadratic","cubic" 为2阶、3阶B样条曲线插值
    if d >= 3 and d+3 < L:
        f = interpolate.interp1d(x, y, kind=kind, fill_value="extrapolate")
        # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
        predict_B_3d_highst.append(max(f(days[d+1]),f(days[d+2]),f(days[d+3])))
predict_B_3d_highst = [0,0,0] + predict_B_3d_highst + [0, 0, 0]

for d in days:
    y = price_gold[0:d+1]
    x = np.linspace(0, d, d+1)
    kind = "cubic"
    # "nearest","zero"为阶梯插值
    # slinear 线性插值
    # "quadratic","cubic" 为2阶、3阶B样条曲线插值
    if d >= 3 and d+3 < L:
        f = interpolate.interp1d(x, y, kind=kind, fill_value="extrapolate")
        # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
        predict_G_3d_highst.append(max(f(days[d+1]),f(days[d+2]),f(days[d+3])))
predict_G_3d_highst = [0,0,0] + predict_G_3d_highst + [0, 0, 0]
'''

#Decision
Tp = 3
alpha_gold = 0.01
alpha_bit = 0.02
R = []
props = []
day = 0
money = 1000
bitcoin = 0
gold = 0
C = []
G = []
B = []
C.append(1000)
G.append(0)
B.append(0)
wait_days = 30

while(day+Tp < L):
    GoldTrade = not (day % 7 == 0 or day % 7 == 1)
    b = B[-1]
    c = C[-1]
    g = G[-1]
    if day < wait_days:
        C.append(c)
        G.append(g)
        B.append(b)
        prop = 1000
        props.append(prop)
        day += 1
        continue
    MG = price_gold[day]
    MB = price_bit[day]
    maxMG = predict_G_3d_highst[day+Tp]
    maxMB = predict_B_3d_highst[day+Tp]
    '''
    for d in range(day+1, day+Tp+1):
        if maxMG < price_gold[d]:
            maxMG = price_gold[d]
        if maxMB < price_bit[d]:
            maxMB = price_bit[d]
    '''
    incG = (maxMG - MG) / MG
    incB = (maxMB - MB) / MB
    newRG = incG - alpha_gold * (2 + incG)
    newRB = incB - alpha_bit * (2 + incB)

    b = B[-1]
    c = C[-1]
    g = G[-1]
    newRB = -10
    if C[-1] > 0:  # C->G/B
        if newRB > newRG and newRB > 0:  # C->B
            b = C[-1] * (1 - alpha_bit) / MB
            c = 0
            g = 0
            print("{}: C->B".format(day))
        elif GoldTrade and newRG > newRB and newRG > 0:  # C->G
            b = 0
            g = C[-1] * (1 - alpha_gold) / MG
            c = 0
            print("{}: C->G".format(day))
        # else:
        # print(newRG)
    elif GoldTrade and G[-1] > 0:  # G->C/B:
        if newRB - newRG > 2 * alpha_gold and newRB > -2 * alpha_bit:  # G->B
            c = 0
            b = G[-1] * (1 - alpha_bit) * MG / MB
            g = 0
            print("{}: G->B".format(day))
        elif newRG < -2 * alpha_gold:  # G->C
            c = G[-1] * (1 - alpha_gold) * MG
            b = 0
            g = 0
            print("{}: G->C".format(day))

    elif B[-1] > 0:  # B->C/G:
        if GoldTrade and newRG - newRB > 2 * alpha_bit and newRG > -2 * alpha_gold:  # B->G
            c = 0
            g = B[-1] * (1 - alpha_gold) * MB / MG
            b = 0
            print("{}: B->G".format(day))
        elif newRB < -2 * alpha_bit:  # B->C
            c = B[-1] * (1 - alpha_bit) * MB
            b = 0
            g = 0
            print("{}: B->C".format(day))
    C.append(c)
    G.append(g)
    B.append(b)
    prop = C[-1] + G[-1] * MG + B[-1] * MB
    props.append(prop)
    day += 1

date = range(0, L-Tp)
plt.style.use('seaborn')
fig, ax =plt.subplots()
#lt.plot(date, predict_B_3d_highst[0:L-Tp],linewidth = 0.6,  c='green')
#plt.plot(date, R, linewidth = 0.3, c='blue')
plt.plot(date, props[0:L-Tp], linewidth = 0.4, c='red')

ax.set_ylabel("Ideal Daily Assets (Tp=3)", fontsize = 16)
ax.set_xlabel("day", fontsize = 16)
print(props[-1])
plt.show()

