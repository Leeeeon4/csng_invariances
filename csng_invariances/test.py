from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time


def test_multiprocessing(args):
    time.sleep(args)


n = range(30)
t = 1


t1 = time.time()

t2 = time.time()

with ProcessPoolExecutor() as executor:
    tqdm(executor.map(test_multiprocessing, [t for _ in n]), total=len(n))

t3 = time.time()

print(f"single process: {round(t2-t1,2)} s\nmultiprocess: {round(t3-t2,2)} s")
