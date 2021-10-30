import torch
import torch.nn.functional as tf
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys

'''人工手写体识别，学习DataLoader加载数据'''


# lr取0.001，能达到98%的准确率

def OneHot(InputTensor):
    """将数字变成独热编码"""

    out = torch.zeros((InputTensor.shape[0]), 10)
    for row in range(len(out)):
        out[row][InputTensor[row]] = 1
    return out


class TextModel(torch.nn.Module):
    """输入N*784，输出N*10"""

    def __init__(self):
        super(TextModel, self).__init__()
        # 784 --> 512
        self.rote11 = torch.nn.Sequential(torch.nn.Linear(784, 1024),
                                          torch.nn.Sigmoid(),
                                          torch.nn.Linear(1024, 512))
        self.rote12 = torch.nn.Linear(784, 512)
        # 512 --> 64
        self.rote21 = torch.nn.Sequential(torch.nn.Linear(512, 256),
                                          torch.nn.Sigmoid(),
                                          torch.nn.Linear(256, 64))
        self.rote22 = torch.nn.Linear(512, 64)
        # 64 --> 10
        self.rote31 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x1 = tf.relu(self.rote11(x) + self.rote12(x))
        x2 = tf.relu(self.rote21(x1) + self.rote22(x1))
        return self.rote31(x2)


def Test_Accuracy():
    """进行测试"""
    correct = 0
    for _, data in enumerate(test_loader):
        TestData = torch.reshape(data[0], (-1, 28 ** 2))
        TestPredLabel = model(TestData)
        for row in range(len(TestPredLabel)):
            if (torch.argmax(TestPredLabel[row, :]) == data[1][row]) == torch.tensor(True):
                correct = correct + 1
    return correct / 10000


if __name__ == "__main__":
    '''加载数据，放于train_loader ，test_loader中'''
    train_dataset = datasets.MNIST(root='/home/txy/zhangzhao/Pytorch_learn/Mnist_learn/mnist',
                                   train=True, transform=transforms.ToTensor(), download=True)
    test_set = datasets.MNIST(root='/home/txy/zhangzhao/Pytorch_learn/Mnist_learn/mnist',
                              train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=32)
    test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=32)
    del train_dataset, test_set

    '''准备神经网络相应参数'''
    model = TextModel()
    LearningRate = 0.0005
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LearningRate)
    criterion = torch.nn.MSELoss()
    epochs = 20

    '''进行训练'''
    Loss = 10
    for epoch in range(epochs):  # 外层整体循环
        for _, data in enumerate(train_loader):  # 内层loader循环
            TrainData = torch.reshape(data[0], (-1, 28 ** 2))
            TrainLabel = OneHot(data[1])
            PredLabel = model(TrainData)

            Loss = criterion(PredLabel, TrainLabel)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
        print('第', epoch, '次损失值为', Loss)
        if epoch % 5 == 0:
            print('在测试上准确度为', 100 * Test_Accuracy(), '%')

    # torch.save(model, '/home/txy/zhangzhao/Pytorch_learn/Mnist_learn/model.pkl')
