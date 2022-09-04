import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data

if '__main__' == __name__:
    data_csv1 = pd.read_csv('./a.csv', usecols=[1])

    # 数据预处理
    data_csv1 = data_csv1.dropna()  # 滤除缺失数据
    dataset1 = data_csv1.values   # 获得csv的值
    dataset1 = dataset1.astype('float32')

    max_value1 = np.max(dataset1)  # 获得最大值
    min_value1 = np.min(dataset1)  # 获得最小值
    scalar1 = max_value1 - min_value1  # 获得间隔数量

    dataset1 = list(map(lambda x: x / scalar1, dataset1))  # 归一化
    print(dataset1)

    data_csv2 = pd.read_csv('./a.csv', usecols=[2])
    data_csv2 = data_csv2.dropna()  # 滤除缺失数据
    dataset2 = data_csv2.values  # 获得csv的值
    dataset2 = dataset2.astype('float32')
    max_value2 = np.max(dataset2)  # 获得最大值
    min_value2 = np.min(dataset2)  # 获得最小值
    # print("max value:", max_value2, "min value:", min_value2)
    scalar2 = max_value2 - min_value2  # 获得间隔数量
    dataset2 = list(map(lambda x: x / scalar2, dataset2))  # 归一化
    # print(dataset2)

    LookBacks = 7
    BATCH_SIZE = 16
    EPOCH = 50
    Tp = 10
    previousPeriod = 30
    Today = 150
    LEN = len(data_csv1)

    def create_dataset(dataset, look_back=LookBacks, predictionTp=Tp, Today=Today):
        dataX, dataY, testX = [], [], []
        for i in range(Today-previousPeriod, Today-look_back-predictionTp):
            a = []
            for item in dataset[i:(i+look_back)]:
                a.append(item)
            dataX.append(a)
            testX = dataset[Today-look_back:Today]

            # a = dataset1[i:(i + look_back)]
            # print("current a1 is :", a, "+", len(a))
            # for item in dataset2[int(datasetBitcoin.iloc[Today-501]+i-Today+previousPeriod):int(datasetBitcoin.iloc[Today-501]+i-Today+previousPeriod+look_back)]:
            #     a.append(item)
            # a = dataset2[int(datasetBitcoin.iloc[Today-501]+i-Today+previousPeriod):int(datasetBitcoin.iloc[Today-501]+i-Today+previousPeriod+look_back)]
            # dataX.extend(a)
            # print("current a2 is :", a, "+", len(a))
            # print("current dataX is :", dataX, len(dataX))

            b = []
            for item in dataset[(i+look_back):(i+look_back+predictionTp)]:
                b.append(item)
            dataY.append(b)

        return np.array(dataX), np.array(dataY), np.array(testX)

    # 创建好输入输出
    # 每次只是创建了该天训练的训练集

    class lstm(nn.Module):
        def __init__(self, input_size=LookBacks, hidden_size=18, output_size=Tp, num_layer=2):
            super(lstm, self).__init__()
            self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, dropout=0.5)
            self.layer2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x, _ = self.layer1(x)
            s, b, h = x.size()
            x = x.view(s * b, h)
            x = self.layer2(x)
            x = x.view(s, b, -1)
            return x

    newLSTM = lstm()
    newLSTM.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(newLSTM.parameters(), lr=0.005)

    prediction = np.zeros((LEN, Tp))
    for day in range(Today, LEN-Tp):
        data_X, data_Y, testX = create_dataset(dataset1, LookBacks, Tp, day)

        train_X = data_X.reshape(-1, 1, LookBacks)
        train_Y = data_Y.reshape(-1, 1, Tp)
        testX = testX.reshape(-1, 1, LookBacks)
        train_X = list(train_X[:])
        train_Y = list(train_Y[:])
        testX = list(testX)
        train_X = torch.tensor(train_X)
        train_Y = torch.tensor(train_Y)
        testX = torch.tensor(testX)

        # train_X = torch.unsqueeze(train_X, dim=1).type(torch.FloatTensor)
        # train_Y = torch.unsqueeze(train_Y, dim=1).type(torch.FloatTensor)
        # train_x = torch.from_numpy(train_X)
        # train_y = torch.from_numpy(train_Y)
        # torch_dataset = Data.TensorDataset(train_X, train_Y)
        # loader = Data.DataLoader(
        #     dataset=torch_dataset,
        #     batch_size=BATCH_SIZE,
        #     shuffle=True,
        #     num_workers=4
        # )

        newLSTM.train()
        for epoch in range(EPOCH):
            # for step, (batch_x, batch_y) in enumerate(loader):
            var_x = Variable(train_X).cuda()
            var_y = Variable(train_Y).cuda()
            # 前向传播
            out = newLSTM(var_x)
            loss = criterion(out, var_y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 25 == 0:  # 每 50 次输出结果
                print('Today is :', day, 'Epoch: {}, Loss: {:.5f}'.format(epoch + 1, loss.item()))

        newLSTM.eval()
        # test_x = train_X[-1]
        # test_x = torch.unsqueeze(test_x, dim=1).type(torch.FloatTensor)
        testX = Variable(testX).cuda()
        predictY = newLSTM(testX)
        predY = predictY.cpu().data.numpy()
        print("Current prediction answer is :", predY)
        predY = predY*(max_value1-min_value1)
        prediction[day] = predY
        print(prediction[day])
        outputPred = pd.DataFrame(prediction)
        outputPred.to_csv('./LSTMFindTpGold.csv', sep=',', header=0, index=0)


    # torch.save(model, 'modelTest.pkl')
    # torch.load('model.pkl')
    # model.eval()  # 转换成测试模式
    # data_X = data_X.reshape(-1, 1, LookBacks)
    # data_X = torch.from_numpy(data_X)
    # var_data = Variable(data_X).cuda()
    # pred_test = model(var_data)  # 测试集的预测结果
    # pred_test.cpu()
    # # 改变输出的格式
    # pred_test = pred_test.view(-1).data.cpu().numpy()
    #
    # plt.plot(pred_test, 'r', label='prediction')
    # plt.plot(dataset, 'b', label='real')
    # pred_test.to_csv('./Prediction', sep=',', header=0, index=0)
    # plt.legend(loc='best')
    # plt.show()



