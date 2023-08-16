from config.configs import get_configs
import pdb
import numpy as np
import random
import os
import argparse
from tqdm import tqdm
from sklearn.metrics.pairwise import paired_distances
from gestureknn.main import gestureknn

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = get_configs()

    print("args", args)

    gestureknn(args)
