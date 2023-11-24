import os
import numpy as np
import matplotlib.pyplot as plt
from main import get_layout_npz_dataset
import tensorflow as tf


plt.style.use("ggplot")

def main():
    # change data path if needed
    home_dir = os.path.expanduser("~")
    
    runtimes = []
    # loop through the xla/default/train directory
    for filename in os.listdir(os.path.join(home_dir, "data/tpugraphs/npz/layout/xla/default/train")):
        # load the file
        d = dict(np.load(
            os.path.join(home_dir,
            "data/tpugraphs/npz/layout/xla/default/train/" + filename
        )))

        runtimes.extend(d["config_runtime"])
    
    print(runtimes)
    runtimes = sorted(runtimes)
    plt.title("XLA Default Runtime Distribution")
    plt.ylabel("Runtime (ns)")
    plt.plot(range(1, len(runtimes) + 1), runtimes)
    plt.show()


def single():
    home_dir = os.path.expanduser("~")
    (layout_train_ds, layout_valid_ds), layout_npz_dataset = get_layout_npz_dataset()

    graph_batch, config_runtimes = next(iter(layout_train_ds.take(1)))
    print(config_runtimes)


if __name__ == "__main__":
    single()
