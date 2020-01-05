import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 8, 5, 2)
        self.relu_conv1_1 = nn.PReLU()
        self.pool1 = nn.AvgPool2d(2, 2, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(8, 16, 3, 1)
        self.relu_conv2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(16, 16, 3, 1)
        self.relu_conv2_2 = nn.PReLU()
        self.pool2 = nn.AvgPool2d(2, 2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(16, 24, 3, 1)
        self.relu_conv3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1)
        self.relu_conv3_2 = nn.PReLU()
        self.pool3 = nn.AvgPool2d(2, 2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        self.relu_conv4_1 = nn.PReLU()
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        self.relu_conv4_2 = nn.PReLU()

        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.relu_ip1 = nn.PReLU()

        self.ip2 = nn.Linear(128, 128)
        self.relu_ip2 = nn.PReLU()

        self.ip3 = nn.Linear(128, 42)

        self.conv4_2_cls = nn.Conv2d(40, 40, 3, 1, 1)
        self.relu_conv4_2_cls = nn.PReLU()

        self.ip1_cls = nn.Linear(4 * 4 * 40, 128)
        self.relu_ip1_cls = nn.PReLU()

        self.ip2_cls = nn.Linear(128, 128)
        self.relu_ip2_cls = nn.PReLU()

        self.ip3_cls = nn.Linear(128, 2)
        self.face_score = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu_conv1_1(self.conv1_1(x)))
        x = self.pool2(self.relu_conv2_2(self.conv2_2(self.relu_conv2_1(self.conv2_1(x)))))
        x = self.pool3(self.relu_conv3_2(self.conv3_2(self.relu_conv3_1(self.conv3_1(x)))))
        x_cls = x = self.relu_conv4_1(self.conv4_1(x))

        x = self.relu_conv4_2(self.conv4_2(x))
        x = x.view(-1, 4 * 4 * 80)
        x = self.relu_ip1(self.ip1(x))
        x = self.ip3(self.relu_ip2(self.ip2(x)))

        x_cls = self.relu_conv4_2_cls(self.conv4_2_cls(x_cls))
        x_cls = x_cls.view(-1, 4 * 4 * 40)
        x_cls = self.relu_ip1_cls(self.ip1_cls(x_cls))
        x_cls = self.ip3_cls(self.relu_ip2_cls(self.ip2_cls(x_cls)))
        x_cls = self.face_score(x_cls)

        return x_cls, x
