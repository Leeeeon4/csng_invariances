import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path_1", type=str, help="increase output verbosity")
parser.add_argument("--path_2", type=str, help="increase output verbosity")
args = parser.parse_args()


def func1(path_1, path_2, **kwargs):
    print("this is func1")
    print(path_1)


def func2(path_1, path_2, **kwargs):
    print("this is func2")
    print(path_2)


func1(**vars(args))
func2(**vars(args))
