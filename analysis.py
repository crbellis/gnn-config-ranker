import os
import json
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


def view_tile_train_run():
    import gzip
    with gzip.open("tile/final/run_.jsonz", 'rb') as f:
        json_bytes = f.read()

    json_str = json_bytes.decode('utf-8')
    data = json.loads(json_str)

    plt.plot(data["train_curve"]["epoch"], data["train_curve"]["train_opa"], label="Training OPA")
    plt.plot(data["train_curve"]["epoch"], data["train_curve"]["val_opa"], label="Validation OPA")
    plt.legend()
    plt.title("Tile Training OPA")
    plt.ylabel("OPA")
    plt.xlabel("Epoch")
    plt.show()

    plt.plot(data["train_curve"]["epoch"], data["train_curve"]["train_loss"], label="Training Loss")
    plt.plot(data["train_curve"]["epoch"], data["train_curve"]["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Tile Training Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()


def get_tile_avg_global():
    # data = json.load(open("tile/errors_val_baseline_full.json"))
    data = json.load(open("layout/errors_val_baseline_full.json"))
    sum_values = 0
    length = 0
    for v in data["1"]:
        sum_values += sum(data["1"][v])
        length += len(data["1"][v])
    
    print(sum_values / length)
    # 0.09586657355224038
    # baseline on filtered ds: 0.13485408708165272


if __name__ == "__main__":
    get_tile_avg_global()