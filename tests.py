import math
import time
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


torch.set_num_threads(torch.get_num_threads())


def start(device, computations, test_int8, test_int16, test_int32, test_int64, test_float32, test_float64, test_nn):
     # create tester object
    tester = Tester(device)

    if test_int8:
        int8_time = tester.test_int8(computations)
        print('int8 time: {:7.1f}s'.format(int8_time))

    if test_int16:
        int16_time = tester.test_int16(computations)
        print('int16 time: {:6.1f}s'.format(int16_time))

    if test_int32:
        int32_time = tester.test_int32(computations)
        print('int32 time: {:6.1f}s'.format(int32_time))

    if test_int64:
        int64_time = tester.test_int64(computations)
        print('int64 time: {:6.1f}s'.format(int64_time))

    if test_float32:
        float32_time = tester.test_float32(computations)
        print('float32 time: {:4.1f}s'.format(float32_time))

    if test_float64:
        float64_time = tester.test_float64(computations)
        print('float64 time: {:4.1f}s'.format(float64_time))

    if test_nn:
        nn_time = tester.test_nn()
        print('nn time: {:9.1f}s'.format(nn_time))


class Tester():
    def __init__(self, device):
        self.device = torch.device('cuda' if device == 'gpu' else 'cpu')

    def test_int8(self, computations):
        axis_size = int(math.pow(computations, 0.25))
        x = torch.randint(low=-11, high=11, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int8)
        m = torch.randint(low=-11, high=11, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int8)

        start = time.time()
        a = torch.matmul(x, m).to(self.device)
        return time.time() - start


    def test_int16(self, computations):
        axis_size = int(math.pow(computations, 0.25))
        x = torch.randint(low=-181, high=181, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int16)
        m = torch.randint(low=-181, high=181, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int16)

        start = time.time()
        a = torch.matmul(x, m).to(self.device)
        return time.time() - start

    def test_int32(self, computations):
        axis_size = int(math.pow(computations, 0.25))
        x = torch.randint(low=-46340, high=46340, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int32)
        m = torch.randint(low=-46340, high=46340, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int32)

        start = time.time()
        a = torch.matmul(x, m).to(self.device)
        return time.time() - start

    def test_int64(self, computations):
        axis_size = int(math.pow(computations, 0.25))
        
        x = torch.randint(low=-2147483648, high=2147483648, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int64)
        m = torch.randint(low=-2147483648, high=2147483648, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int64)

        start = time.time()
        a = torch.matmul(x, m).to(self.device)
        return time.time() - start

    def test_float32(self, computations):
        axis_size = int(math.pow(computations, 0.25))
        x = torch.rand(size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.float32)
        m = torch.rand(size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.float32)

        start = time.time()
        a = torch.matmul(x, m).to(self.device)
        return time.time() - start

    def test_float64(self, computations):
        axis_size = int(math.pow(computations, 0.25))
        x = torch.rand(size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.float64)
        m = torch.rand(size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.float64)

        start = time.time()
        a = torch.matmul(x, m).to(self.device)
        return time.time() - start

    def test_nn(self):
        transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                shuffle=True, num_workers=2)

        net=Net()
        net.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        start = time.time()
        for epoch in range(50):  # loop over the dataset multiple times
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return time.time() - start


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 200, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(200, 200, 5)
        self.fc1 = torch.nn.Linear(200 * 5 * 5, 1000)
        self.fc2 = torch.nn.Linear(1000, 1000)
        self.fc3 = torch.nn.Linear(1000, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 200 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x