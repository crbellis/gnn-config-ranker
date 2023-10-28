import tensorflow as tf
import tensorflow_gnn as tfgnn
from utils import implicit

def _mlp(dims, hidden_activation, l2reg=1e-4, use_bias=True):
  """Helper function for multi-layer perceptron (MLP)."""
  layers = []
  for i, dim in enumerate(dims):
    if i > 0:
      layers.append(tf.keras.layers.Activation(hidden_activation))
    layers.append(tf.keras.layers.Dense(
        dim, kernel_regularizer=tf.keras.regularizers.l2(l2reg),
        use_bias=use_bias))
  return tf.keras.Sequential(layers)


class _OpEmbedding(tf.keras.Model):
  """Embeds GraphTensor.node_sets['op']['op'] nodes into feature 'op_e'."""

  def __init__(self, num_ops: int, embed_d: int, l2reg: float = 1e-4):
    super().__init__()
    self.embedding_layer = tf.keras.layers.Embedding(
        num_ops, embed_d, activity_regularizer=tf.keras.regularizers.l2(l2reg))

  def call(
      self, graph: tfgnn.GraphTensor,
      training: bool = False) -> tfgnn.GraphTensor:
    op_features = dict(graph.node_sets['op'].features)
    op_features['op_e'] = self.embedding_layer(
        tf.cast(graph.node_sets['op']['op'], tf.int32))
    return graph.replace_features(node_sets={'op': op_features})


class ResModel(tf.keras.Model):
    """GNN with residual connections."""

    def __init__(
        self, num_configs: int, num_ops: int, op_embed_dim: int = 32,
        num_gnns: int = 2, mlp_layers: int = 2,
        hidden_activation: str = 'leaky_relu',
        hidden_dim: int = 32, reduction: str = 'sum'):
        super().__init__()
        self._num_configs = num_configs
        self._num_ops = num_ops
        self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
        self._prenet = _mlp([hidden_dim] * mlp_layers, hidden_activation)
        self._gc_layers = []
        for _ in range(num_gnns):
            self._gc_layers.append(_mlp([hidden_dim] * mlp_layers, hidden_activation))
        self._postnet = _mlp([hidden_dim, 1], hidden_activation, use_bias=False)

    def call(self, graph: tfgnn.GraphTensor, training: bool = False):
        del training
        return self.forward(graph, self._num_configs)

    def _node_level_forward(
        self, node_features: tf.Tensor,
        config_features: tf.Tensor,
        graph: tfgnn.GraphTensor, num_configs: int,
        edgeset_prefix='') -> tf.Tensor:
        adj_op_op = implicit.AdjacencyMultiplier(
            graph, edgeset_prefix+'feed')  # op->op
        adj_config = implicit.AdjacencyMultiplier(
            graph, edgeset_prefix+'config')  # nconfig->op

        adj_op_op_hat = (adj_op_op + adj_op_op.transpose()).add_eye()
        adj_op_op_hat = adj_op_op_hat.normalize_symmetric()

        x = node_features

        x = tf.stack([x] * num_configs, axis=1)
        config_features = 100 * (adj_config @ config_features)
        x = tf.concat([config_features, x], axis=-1)
        x = self._prenet(x)
        x = tf.nn.leaky_relu(x)

        for layer in self._gc_layers:
            y = x
            y = tf.concat([config_features, y], axis=-1)
            y = tf.nn.leaky_relu(layer(adj_op_op_hat @ y))
            x += y
        return x

    def forward(
        self, graph: tfgnn.GraphTensor, num_configs: int,
        backprop=True) -> tf.Tensor:
        graph = self._op_embedding(graph)

        config_features = graph.node_sets['nconfig']['feats']
        node_features = tf.concat([
            graph.node_sets['op']['feats'],
            graph.node_sets['op']['op_e']
        ], axis=-1)

        x_full = self._node_level_forward(
            node_features=tf.stop_gradient(node_features),
            config_features=tf.stop_gradient(config_features),
            graph=graph, num_configs=num_configs)

        if backprop:
            x_backprop = self._node_level_forward(
                node_features=node_features,
                config_features=config_features,
                graph=graph, num_configs=num_configs, edgeset_prefix='sampled_')

            is_selected = graph.node_sets['op']['selected']
            # Need to expand twice as `is_selected` is a vector (num_nodes) but
            # x_{backprop, full} are 3D tensors (num_nodes, num_configs, num_feats).
            is_selected = tf.expand_dims(is_selected, axis=-1)
            is_selected = tf.expand_dims(is_selected, axis=-1)
            x = tf.where(is_selected, x_backprop, x_full)
        else:
            x = x_full

        adj_config = implicit.AdjacencyMultiplier(graph, 'config')

        # Features for configurable nodes.
        config_feats = (adj_config.transpose() @ x)

        # Global pooling
        adj_pool_op_sum = implicit.AdjacencyMultiplier(graph, 'g_op').transpose()
        adj_pool_op_mean = adj_pool_op_sum.normalize_right()
        adj_pool_config_sum = implicit.AdjacencyMultiplier(
            graph, 'g_config').transpose()
        x = self._postnet(tf.concat([
            # (A D^-1) @ Features
            adj_pool_op_mean @ x,
            # l2_normalize( A @ Features )
            tf.nn.l2_normalize(adj_pool_op_sum @ x, axis=-1),
            # l2_normalize( A @ Features )
            tf.nn.l2_normalize(adj_pool_config_sum @ config_feats, axis=-1),
        ], axis=-1))

        x = tf.squeeze(x, -1)

        return x