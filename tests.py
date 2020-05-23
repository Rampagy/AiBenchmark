import math
import time
import torch



def start(device, computations, test_int8, test_int16, test_int32, test_int64, test_float32, test_float64, test_nn):
    overall_start = time.time()

    # create tester object
    tester = Tester(device)

    if test_int8:
        int8_time = tester.test_int8(computations)
        print(int8_time)



    print(time.time() - overall_start)


class Tester():
    def __init__(self, device):
        self.device = torch.device('cuda' if device == 'gpu' else 'cpu')

    def test_int8(self, computations):
        int8_start = time.time()

        x = torch.randint(low=0, high=16, size=(computations, 1), dtype=torch.int8)
        m = torch.randint(low=0, high=16, size=(1, computations), dtype=torch.int8)
        torch.mm(x, m).to(self.device)

        return (time.time() - int8_start)


    def test_int16():
        input_tensor = torch.randn((computations, 1), dtype=torch.int16)

    def test_int32():
        input_tensor = torch.randn((computations, 1), dtype=torch.int32)

    def test_int64():
        input_tensor = torch.randn((computations, 1), dtype=torch.int64)

    def test_float32():
        input_tensor = torch.randn((computations, 1), dtype=torch.float32)

    def test_float64():
        input_tensor = torch.randn((computations, 1), dtype=torch.float64)

    def test_nn():
        pass