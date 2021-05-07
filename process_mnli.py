import sys
import multiprocessing
import os
from multiprocessing import Pool
import subprocess
from math import ceil

def proving(startnum, endnum):
    process = subprocess.Popen(["./scripts/eval_mnli_train.sh", "gq", "1", "scripts/semantic_templates.yaml", str(startnum), str(endnum)])
    process.wait()
    print(f"Finishing {startnum} {endnum}")


if __name__ == '__main__':
    
    start = int(sys.argv[1])
    number = int(sys.argv[2])
    ncore = multiprocessing.cpu_count()
    print("ncore", ncore)
    execute_number = int(ceil(float(number - start) / ncore))
    with Pool(ncore) as p:
        end = number
        print("process", [(i, i+execute_number-1) for i in range(start, end+1, execute_number)])
        p.starmap(proving, [(i, i+execute_number-1) for i in range(start, end+1, execute_number)])
