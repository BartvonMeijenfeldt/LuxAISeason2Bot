import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()

df = pd.read_json(args.path)
duration = df.apply(lambda x: x[0]["duration"], axis=1)
duration = np.clip(duration - 3, a_min=0, a_max=np.inf)
print(duration.sum())
