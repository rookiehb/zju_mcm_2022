import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if '__main__' == __name__ :

    LR = 0.0001
    BATCH_SIZE = 16
    EPOCH = 1

    path = 'flights.csv'
    data1 = pd.read_csv(path)
    npdata = np.array(data1)

    data = torch.tensor(npdata)

    features = data[:, 0].long()
    label = data[:, -1].long()

    train_size = int(len(features) * 0.9)
    test_size = len(features) - train_size
    train_X = features[:train_size]
    train_Y = label[:train_size]
    test_X = features[train_size:]
    test_Y = label[train_size:]

    x = torch.unsqueeze(train_X, dim=1).type(torch.FloatTensor)
    y = torch.unsqueeze(train_Y, dim=1).type(torch.FloatTensor)

    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)     # 隐藏层 输出n_features,输出n_hidden
            self.predict = torch.nn.Linear(n_hidden, n_output)
            self.dropout = torch.nn.Dropout(0.3)

        def forward(self, x):
            x = self.hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
            # x = self.hidden(x)
            # x = self.dropout(x)
            # x = F.relu(x)
            # x = self.hidden(x)
            # x = self.dropout(x)
            # x = F.relu(x)
            x = self.predict(x)   # 回归函数的时候可以不用激励函数
            return x

    net = Net(1, 48, 1)
    print(net)
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_func = torch.nn.MSELoss()      # 均方差

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)

            output = net(b_x)
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch:", epoch, "|step:", step, "|loss:", loss)

    torch.save(net, 'net.pkl')
    net.eval()

    test = torch.unsqueeze(test_X, dim=1).type(torch.FloatTensor)

    # test_data = Variable(data_X).cudu()
    #
    # data_X = data_X.reshape(-1, 1, LookBacks)

    pred_test = net(test)
    print(pred_test)
    # data_X = torch.from_numpy(data_X)
    # var_data = Variable(data_X).cuda()
    # pred_test = model(var_data)  # 测试集的预测结果
    # pred_test.cpu()
    # 改变输出的格式
    # pred_test = pred_test.view(-1).data.cpu().numpy()

    plt.plot(pred_test.data.numpy(), 'r', label='prediction')
    plt.plot(features, 'b', label='real')
    plt.legend(loc='best')
    plt.show()

