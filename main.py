import os
import collections
from typing import Tuple
import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_ranking as tfr
import utils.layout_data as layout_data
import utils.implicit as implicit
import utils.tile_data as tile_data
from model import _OpEmbedding, ResModel
import functools
import json

home_dir = os.path.expanduser("~")
LAYOUT_DATA_ROOT = os.path.join(home_dir, 'data/tpugraphs/npz/layout')
TILE_DATA_ROOT = os.path.join(home_dir, "data/tpugraphs/npz/tile")
SOURCE = 'xla'  # Can be "xla" or "nlp"
SEARCH = 'random'  # Can be "random" or "default"

# Batch size information.
BATCH_SIZE = 16  # Number of graphs per batch.
CONFIGS_PER_GRAPH = 5  # Number of configurations (features and target values) per graph.
MIN_TRAIN_CONFIGS = 5
MAX_TRAIN_CONFIGS = 1000
MAX_KEEP_NODES = 1000  # Useful for dropout.
# `MAX_KEEP_NODES` is (or, is not) useful for Segment Dropout, if model uses
# edges "sampled_config" and "sampled_feed" (or, "config" and "feed")

def main():
    home_dir = os.path.expanduser("~")

    (layout_train_ds, layout_valid_ds), layout_npz_dataset = get_layout_npz_dataset()
    print("loaded")
    # (tile_train_ds, tile_valid_ds) = get_tile_npz_dataset()

    graph_batch, config_runtimes = next(iter(layout_train_ds.take(2)))
    
    # print('graph_batch = ')
    # print(graph_batch)
    # print('\n\n')
    print('config_runtimes=')
    print(config_runtimes)
    print(layout_train_ds.cardinality().numpy() * BATCH_SIZE * CONFIGS_PER_GRAPH)

    # model = ResModel(CONFIGS_PER_GRAPH, layout_npz_dataset.num_ops, op_embed_dim=16, hidden_dim=128)
    model = ResModel(CONFIGS_PER_GRAPH, layout_npz_dataset.num_ops, op_embed_dim=16, hidden_dim=128)

    loss = tfr.keras.losses.ListMLELoss()  # (temperature=10)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, clipnorm=0.5)

    model.compile(loss=loss, optimizer=opt, metrics=[
        tfr.keras.metrics.OPAMetric(name='opa_metric'),
    ])

    early_stop = 20 # If validation OPA did not increase in this many epochs, terminate training.
    best_params = None  # Stores parameters corresponding to best validation OPA, to restore to them after training.
    best_val_opa = -1  # Tracks best validation OPA
    best_val_at_epoch = -1  # At which epoch.
    epochs = 50 # Total number of training epochs.
    val_losses = []
    train_losses = []
    train_opas = []
    val_opas = []

    try:
        for i in range(epochs):
            print("epoch: ", i + 1)
            history = model.fit(
                layout_train_ds, epochs=1, verbose=1, validation_data=layout_valid_ds,
                validation_freq=1)

            train_loss = history.history['loss'][-1]
            train_opa = history.history['opa_metric'][-1]
            val_loss = history.history['val_loss'][-1]
            val_opa = history.history['val_opa_metric'][-1]
            val_losses.append(val_loss)
            train_losses.append(train_loss)
            train_opas.append(train_opa)
            val_opas.append(val_opa)
            if val_opa > best_val_opa:
                best_val_opa = val_opa
                best_val_at_epoch = i
                best_params = {v.ref: v + 0 for v in model.trainable_variables}
                print(' * [@%i] Validation (NEW BEST): %s' % (i+1, str(val_opa)))
            elif early_stop > 0 and i - best_val_at_epoch >= early_stop:
                print('[@%i] Best accuracy was attained at epoch %i. Stopping.' % (i+1, best_val_at_epoch))
                break

    except Exception:
        pass
    
    json.dump(
        {
            "val_loss": val_losses,
            "train_loss": train_losses,
            "train_opa": train_opas,
            "val_opas": val_opas
        },
        open(f"metrics_{epochs}.json", "w+")
    )

    # Restore best parameters.
    print('Restoring parameters corresponding to the best validation OPA.')
    assert best_params is not None
    for v in model.trainable_variables:
        v.assign(best_params[v.ref])

    model.save(f"layout_model_baseline_{epochs}")

    # model.load_weights("./model_1.h5")

    # test model output on validation sample
    data = layout_valid_ds.take(1)
    for _, y in data:
        print(tf.argsort(y).numpy())
    print(tf.argsort(model.predict(data)).numpy())


def get_layout_npz_dataset() -> Tuple[
    Tuple[tf.data.Dataset, tf.data.Dataset], layout_data.NpzDataset]:
    layout_data_root_dir = os.path.join(
      os.path.expanduser(LAYOUT_DATA_ROOT), SOURCE, SEARCH)

    layout_npz_dataset = layout_data.get_npz_dataset(
        layout_data_root_dir,
        min_train_configs=MIN_TRAIN_CONFIGS,
        max_train_configs=MAX_TRAIN_CONFIGS,  # If any graph has more than this configurations, it will be filtered [speeds up loading + training]
        cache_dir='cache'
    )

    layout_train_ds = (
        layout_npz_dataset.train.get_graph_tensors_dataset(
            CONFIGS_PER_GRAPH, max_nodes=MAX_KEEP_NODES)
        .shuffle(100, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .map(tfgnn.GraphTensor.merge_batch_to_components)
        .map(pair_layout_graph_with_label))

    layout_valid_ds = (
        layout_npz_dataset.validation.get_graph_tensors_dataset(
            CONFIGS_PER_GRAPH)
        .batch(BATCH_SIZE, drop_remainder=False)
        .map(tfgnn.GraphTensor.merge_batch_to_components)
        .map(pair_layout_graph_with_label))

    return (layout_train_ds, layout_valid_ds), layout_npz_dataset


def get_tile_npz_dataset() -> Tuple[
    Tuple[tf.data.Dataset, tf.data.Dataset], tile_data.NpzDataset]:
    tile_data_root_dir = os.path.join(
      os.path.expanduser(TILE_DATA_ROOT), SOURCE)

    tile_npz_dataset = tile_data.get_npz_dataset(
        tile_data_root_dir,
        min_train_configs=MIN_TRAIN_CONFIGS,
        cache_dir='cache'
    )

    map_label_pairs = functools.partial(
        pair_tile_graph_with_label,
        batch_size=BATCH_SIZE, 
        num_configs=CONFIGS_PER_GRAPH
    )

    tile_train_ds = (
        tile_npz_dataset.train.get_graph_tensors_dataset(
            CONFIGS_PER_GRAPH)
      .shuffle(5000, reshuffle_each_iteration=True)
      .batch(BATCH_SIZE, drop_remainder=True)
      .map(tfgnn.GraphTensor.merge_batch_to_components)
      .map(map_label_pairs))

    tile_val_ds = (
        tile_npz_dataset.validation.get_graph_tensors_dataset(
            CONFIGS_PER_GRAPH)
      .shuffle(5000, reshuffle_each_iteration=True)
      .batch(BATCH_SIZE, drop_remainder=True)
      .map(tfgnn.GraphTensor.merge_batch_to_components)
      .map(map_label_pairs))

    return (tile_train_ds, tile_val_ds)


def pair_layout_graph_with_label(graph: tfgnn.GraphTensor):
    """Extracts label from graph (`tfgnn.GraphTensor`) and returns a pair of `(graph, label)`"""
    # Return runtimes divded over large number: only ranking is required. The
    # runtimes are in the 100K range
    label = tf.cast(graph.node_sets['g']['runtimes'], tf.float32) / 1e7
    return graph, label


def pair_tile_graph_with_label(graph: tfgnn.GraphTensor, batch_size=10, num_configs=2):
  label = tf.reshape(
      graph.node_sets['config']['runtimes'], [batch_size, num_configs])
  return graph, label


# Used for validation. For training, data.py accepts `min_train_configs`.
def _graph_has_enough_configs(graph: tfgnn.GraphTensor, num_configs=2):
    """To used to filter validation dataset."""
    return graph.node_sets['config'].sizes[0] >= num_configs

def eval():
    data_root_dir = os.path.join(
        os.path.expanduser("~"), SOURCE, SEARCH)

    (_, val_ds), ds = get_layout_npz_dataset()


if __name__ == "__main__":
    main()