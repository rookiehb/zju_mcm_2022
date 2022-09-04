import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import pylab as pl
from datetime import datetime

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
        #dates.append(current_date)
with open(filename_G) as fg:
    reader = csv.reader(fg)
    header_row = next(reader)

    price_gold = []
    for row in reader:
        single_price_gold = float(row[1])
        price_gold.append(single_price_gold)

Tp = 3
alpha = 0.02
L = len(price_bit)
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

while(day+Tp < L):
    MG = price_gold[day]
    MB = price_bit[day]
    maxMG = 0
    maxMB = 0
    for d in range(day+1, day+Tp+1):
        if maxMG < price_gold[d]:
            maxMG = price_gold[d]
        if maxMB < price_bit[d]:
            maxMB = price_bit[d]
    incG = (maxMG - MG) / MG
    incB = (maxMB - MB) / MB
    newRG = incG - alpha * (2 + incG)
    newRB = incB - alpha*(2 + incB)

    b = B[-1]
    c = C[-1]
    g = G[-1]
    #newRG = -10
    if C[-1] > 0:#C->G/B
        if newRB > newRG and newRB > 0: #C->B
            b = C[-1]*(1-alpha) / MB
            c = 0
            g = 0
            print("{}: C->B".format(day))
        elif newRG > newRB and newRG > 0: #C->G
            b = 0
            g = C[-1] * (1 - alpha) / MG
            c = 0
            print("{}: C->G".format(day))
    elif G[-1] > 0:#G->C/B:
        if newRB - newRG > 2*alpha: #G->B
            c = 0
            b = G[-1]*(1-alpha) * MG / MB
            g = 0
            print("{}: G->B".format(day))
        elif newRG < -2*alpha: #G->C
            c = G[-1] * (1-alpha) * MG
            b = 0
            g = 0
            print("{}: G->C".format(day))

    elif B[-1] > 0:#B->C/G:
        if newRG - newRB > 2*alpha and newRG > -2*alpha: #B->G
            c = 0
            g = B[-1]*(1-alpha) * MB / MG
            b = 0
            print("{}: B->G".format(day))
        elif newRB < -2*alpha: #B->C
            c = B[-1] * (1-alpha) * MB
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
plt.plot(date, props[0:L-Tp], linewidth = 0.4, c='red')
ax.set_ylabel("Ideal Daily Assets (Tp=3)", fontsize = 16)
ax.set_xlabel("day", fontsize = 16)
plt.show()
print("Ideal Final Assets:${}".format(props[-1]))