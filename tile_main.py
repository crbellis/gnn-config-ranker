import functools
import gzip
import io
import json
import os
from typing import Callable, Any
import tensorflow as tf
import tensorflow_gnn as tfgnn
from utils import tile_data as data
import tensorflow_ranking as tfr
from model import _SAGE, _EarlyJoin, _LateJoin, _ResGCN, EarlyJoinSAGE
from utils import metrics
import tqdm


home_dir = os.path.expanduser("~")
BATCH_SIZE = 32
CONFIGS_PER_GRAPH = 2
MIN_TRAIN_CONFIGS = 2
EPOCHS = 100
TILE_DATA_ROOT = os.path.join(home_dir, "data/tpugraphs/npz/tile")
SOURCE = "xla"


def _graph_and_label(graph: tfgnn.GraphTensor, batch_size=10, num_configs=2):
  label = tf.reshape(
      graph.node_sets['config']['runtimes'], [batch_size, num_configs])
  return graph, label


# Used for validation. For training, data.py accepts `min_train_configs`.
def _graph_has_enough_configs(graph: tfgnn.GraphTensor, num_configs=2):
  """To used to filter validation dataset."""
  return graph.node_sets['config'].sizes[0] >= num_configs


def save_model(
    model: tf.keras.Model, run_info: dict[str, Any], out_dir: str,
):
  """Writes `model` and `run_info` onto `out_dir`/*`args.compute_hash()`*."""

  # Save run file.
  out_run_file = os.path.join(out_dir, f'run_.jsonz')
  bytes_io = io.BytesIO()
  with gzip.open(bytes_io, 'wb') as fout:
    fout.write(json.dumps(run_info).encode())
  with tf.io.gfile.GFile(out_run_file, 'wb') as fout:
    fout.write(bytes_io.getvalue())

  # Keras model.
  out_model_file = os.path.join(out_dir, f'model_')
  model.save(out_model_file)


def train():
  """Training loop. `train_args.py` contains description of arguments."""

  # Will be written in out_dir.
  run_info = dict(
      train_curve=dict(
          epoch=[], train_loss=[], train_opa=[], val_loss=[], val_opa=[]),
      final_error=dict(),
  )

  tile_root_dir = os.path.join(os.path.expanduser(TILE_DATA_ROOT), SOURCE)
  # Input training data.
  dataset_partitions = data.get_npz_dataset(
      tile_root_dir, min_train_configs=MIN_TRAIN_CONFIGS,
      cache_dir="cache")
 
  train_ds = (
      dataset_partitions.train.get_graph_tensors_dataset(CONFIGS_PER_GRAPH)
      .shuffle(5000, reshuffle_each_iteration=True)
      .batch(BATCH_SIZE, drop_remainder=True)
      .map(tfgnn.GraphTensor.merge_batch_to_components))

  # Model.
  model = EarlyJoinSAGE(CONFIGS_PER_GRAPH, dataset_partitions.num_ops, hidden_dim=128, op_embed_dim=128)

  # ListMLELoss:1 or MSE:0.02
  loss = metrics.CombinedLoss(metrics.parse_loss_str('ListMLELoss:0.02'))
  opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

  model.compile(loss=loss, optimizer=opt, metrics=[
      tfr.keras.metrics.OPAMetric(name='opa_metric'),
  ])
  attach_labels_fn = functools.partial(
      _graph_and_label, batch_size=BATCH_SIZE, num_configs=CONFIGS_PER_GRAPH)
  train_ds = train_ds.map(attach_labels_fn)

  valid_ds = (
      dataset_partitions.validation.get_graph_tensors_dataset(CONFIGS_PER_GRAPH)
      # Get an extra 5% as we follow with `filter()`.
      .take(int(BATCH_SIZE * 1.05))
      .filter(
          functools.partial(_graph_has_enough_configs, num_configs=CONFIGS_PER_GRAPH))
      .batch(BATCH_SIZE, drop_remainder=True)
      .map(tfgnn.GraphTensor.merge_batch_to_components)
      .map(attach_labels_fn))

  best_params = None
  best_val_opa = -1
  best_val_at_epoch = -1
  train_curve = run_info['train_curve']  # For short.
  for i in range(EPOCHS):
    # old_alsologtostderr = flags.FLAGS.alsologtostderr
    # flags.FLAGS.alsologtostderr = True
    history = model.fit(
        train_ds, epochs=1, verbose=1, validation_data=valid_ds)
    # flags.FLAGS.alsologtostderr = old_alsologtostderr
    train_curve['epoch'].append(i)
    train_curve['train_loss'].append(history.history['loss'][-1])
    train_curve['train_opa'].append(history.history['opa_metric'][-1])
    train_curve['val_loss'].append(history.history['val_loss'][-1])
    train_curve['val_opa'].append(history.history['val_opa_metric'][-1])
    val_opa = history.history['val_opa_metric'][-1]
    if val_opa > best_val_opa:
      best_val_opa = val_opa
      best_val_at_epoch = i
      best_params = {v.ref: v + 0 for v in model.trainable_variables}
      print(' * [@%i] Validation (NEW BEST): %s', i, str(val_opa))
      # Write model and train metrics (in `run_info`).
      save_model(model, run_info, "./tile/")
    # elif args.early_stop > 0 and i - best_val_at_epoch >= args.early_stop:
    #   print('[@%i] Best accuracy was attained at epoch %i. Stopping.',
    #                i, best_val_at_epoch)
    #   break

  # Restore best parameters.
  assert best_params is not None
  for v in model.trainable_variables:
    v.assign(best_params[v.ref])

  # Run on full validation.
  run_info['final_error']['val'] = metrics.top_error_performance(
      dataset_partitions.validation.get_graph_tensors_dataset(), model.forward)

#   # Run on test set.
#   test_ds = dataset_partitions.test.get_graph_tensors_dataset()
# #   if args.test_mode == 'metrics':
# #     run_info['final_error']['test'] = metrics.top_error_performance(
# #         test_ds, model.forward)
# #   elif args.test_mode == 'predictions':
# #     module_ids, ranks = rank_config_indices(test_ds, model.forward)

# #     write_least_runtimes_csv(args.results_csv, module_ids, ranks)

#     ### Add test predictions into run_info file.
#     run_info['test_predictions'] = {}
#     module_ids = module_ids.numpy().tolist()
#     predictions = ranks.numpy().tolist()
#     for module_id, module_predictions in zip(module_ids, predictions):
#       module_id = bytes(module_id).decode()
#       run_info['test_predictions'][module_id] = module_predictions

#   save_model(model, run_info, out_dir, args)


def rank_config_indices(
    test_ds: tf.data.Dataset,
    model_fn: Callable[[tfgnn.GraphTensor, int], tf.Tensor],
    top_ranks=10
    ) -> tuple[tf.Tensor, tf.Tensor]:
  """Module IDs and config indices that `model_fn` assigns lowest scores.

  Args:
    test_ds: Test dataset containing `GraphTensor` instances. Each instance must
      have node sets `'config'` and `'g'` (with feature 'tile_id')
    model_fn: Callable (e.g., tf.Keras model) that will be invoked on every item
      in `test_ds` and the number of configurations (=N). It is expeted to
      return tensor of shape (1, N). The least indices will be output.
    top_ranks: Only this many least indices will be kept.

  Returns:
    Two `tf.Tensor` instances. The first is a vector with entry `[i]` being the
    `graph.node_sets['g']['tile_id']` of the `i`th element of `test_ds`. The
    second is a matrix with width `top_ranks`, where row `[i]` being the least
    `top_ranks` indices when invoking `model_fn` on `graph`.
  """
  all_sorted_indices = []
  all_module_ids = []
  for graph in tqdm.tqdm(test_ds, desc='Generating Predictions'):
    num_configs = int(graph.node_sets['config'].sizes[0].numpy())
    preds = model_fn(graph, num_configs)
    preds = tf.squeeze(preds, 0)  # Remove batch size (of 1)
    sorted_indices = tf.argsort(preds)
    sorted_indices = tf.concat([  # zero-pad.
        sorted_indices, tf.zeros([top_ranks], dtype=sorted_indices.dtype)
    ], axis=0)
    sorted_indices = sorted_indices[:top_ranks]
    all_sorted_indices.append(sorted_indices)
    all_module_ids.append(graph.node_sets['g']['tile_id'][0])

  return tf.stack(all_module_ids, axis=0), tf.stack(all_sorted_indices, axis=0)


def write_least_runtimes_csv(
    out_csv_filepath: str, module_ids: tf.Tensor, ranks: tf.Tensor):
  """Writes CSV file with line `i` containing module_ids[i] and ranks[i]."""
  csv_ranks = tf.strings.join(
      tf.strings.as_string(tf.transpose(ranks)), ';')

  stack_join = lambda x, delim: tf.strings.join(tf.stack(x), delim)
  with tf.io.gfile.GFile(out_csv_filepath, 'w') as fout:
    fout.write('ID,TopConfigs\n')
    id_vector = stack_join(
        [tf.fill(module_ids.shape, 'tile:xla'), module_ids], ':')
    csv_lines = stack_join([id_vector, csv_ranks], ',')
    fout.write(stack_join(csv_lines, '\n').numpy().decode('utf-8'))
  print('\n\n   ***  Wrote', out_csv_filepath, '\n\n')


if __name__ == "__main__":
  train()