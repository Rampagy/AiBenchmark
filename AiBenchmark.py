import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Test devices AI computational capability')

    parser.add_argument('-d', '--device', type=str, default='gpu', help='\'cpu\' or \'gpu\'')
    parser.add_argument('-c', '--computations', type=int, default=1000000000, help='Number of int8, int16, int32, int64, float32, and float64 computations')
    parser.add_argument('-int8', '--test_int8', action="store_true", default=False, help='Test integer 8')
    parser.add_argument('-int16', '--test_int16', action="store_true", default=False, help='Test integer 16')
    parser.add_argument('-int32', '--test_int32', action="store_true", default=False, help='Test integer 32')
    parser.add_argument('-int64', '--test_int64', action="store_true", default=False, help='Test integer 64')
    parser.add_argument('-float32', '--test_float32', action="store_true", default=False, help='Test float 32')
    parser.add_argument('-float64', '--test_float64', action="store_true", default=False, help='Test float 64')
    parser.add_argument('-nn', '--test_nn', action="store_true", default=False, help='Test neural network')
    args = parser.parse_args()

    if args.device in ['cpu', 'gpu']:
        if not (args.test_int8 or 
                args.test_int16 or 
                args.test_int32 or 
                args.test_int64 or 
                args.test_float32 or 
                args.test_float64 or 
                args.test_nn):
            # if all tests are false (none were specified), test them all
            args.test_int8 = True
            args.test_int16 = True
            args.test_int32 = True
            args.test_int64 = True
            args.test_float32 = True
            args.test_float64 = True
            args.test_nn = True

        print(args.device)
        print(args.computations)
        print(args.test_int8)
        print(args.test_int16)
        print(args.test_int32)
        print(args.test_int64)
        print(args.test_float32)
        print(args.test_float64)
        print(args.test_nn)
    else:
        print('Invalid device. Use \'-d gpu\' or \'-d cpu\'')