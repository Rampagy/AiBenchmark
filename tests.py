import math
import time
import torch
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


class Tester():
    def __init__(self, device):
        self.device = torch.device('cuda' if device == 'gpu' else 'cpu')

    def test_int8(self, computations):
        axis_size = int(math.pow(computations, 0.25))
        x = torch.randint(low=-11, high=11, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int8)
        m = torch.randint(low=-11, high=11, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int8)

        start = time.time()
        a = torch.matmul(x, m).to(self.device)
        return (time.time() - start)


    def test_int16(self, computations):
        axis_size = int(math.pow(computations, 0.25))
        x = torch.randint(low=-181, high=181, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int16)
        m = torch.randint(low=-181, high=181, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int16)

        start = time.time()
        a = torch.matmul(x, m).to(self.device)
        return (time.time() - start)

    def test_int32(self, computations):
        axis_size = int(math.pow(computations, 0.25))
        x = torch.randint(low=-46340, high=46340, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int32)
        m = torch.randint(low=-46340, high=46340, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int32)

        start = time.time()
        a = torch.matmul(x, m).to(self.device)
        return (time.time() - start)

    def test_int64(self, computations):
        axis_size = int(math.pow(computations, 0.25))
        
        x = torch.randint(low=-2147483648, high=2147483648, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int64)
        m = torch.randint(low=-2147483648, high=2147483648, size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.int64)

        start = time.time()
        a = torch.matmul(x, m).to(self.device)
        return (time.time() - start)

    def test_float32(self, computations):
        axis_size = int(math.pow(computations, 0.25))
        x = torch.rand(size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.float32)
        m = torch.rand(size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.float32)

        start = time.time()
        a = torch.matmul(x, m).to(self.device)
        return (time.time() - start)

    def test_float64(self, computations):
        axis_size = int(math.pow(computations, 0.25))
        x = torch.rand(size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.float64)
        m = torch.rand(size=(axis_size, axis_size, axis_size, axis_size), dtype=torch.float64)

        start = time.time()
        a = torch.matmul(x, m).to(self.device)
        return (time.time() - start)

    def test_nn():
        pass