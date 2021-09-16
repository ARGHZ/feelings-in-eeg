from concurrent.futures import as_completed, ProcessPoolExecutor
import time
import numpy as np
import winprocess


def do_work(idx1, idx2):
    time.sleep(0.2)
    return np.mean([idx1, idx2])


def run_process_example():
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = set()
        for idx in range(32):
            future = winprocess.submit(
                executor, do_work, idx, idx * 2
            )
            futures.add(future)

        for future in as_completed(futures):
            print(future.result())


if __name__ == '__main__':
    run_process_example()
