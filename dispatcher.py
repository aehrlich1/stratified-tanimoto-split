import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.multiprocessing as mp
import yaml

from main import main


def run_job(dic):
    start = time.time()
    main(dic)
    return f"job done in {time.time() - start:.2f}s"


# --- Simple dispatcher --------------------------------------------------------
def dispatch(jobs, per_run_cpus=1):
    total_cpus = os.cpu_count() or 1
    max_workers = max(1, total_cpus // per_run_cpus)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_job, dic) for dic in jobs]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


# --- Example usage ------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.set_num_threads(1)

    with open("hyperparams.yml", "r") as f:
        params = yaml.safe_load(f)

        keys = params.keys()
        values = params.values()

        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(f"Number of configurations: {len(combinations)}")

        outputs = dispatch(combinations, per_run_cpus=1)
        for line in outputs:
            print(line)
