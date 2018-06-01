# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Training harness for MobileNet v1.

This demonstrates how to train the Mobilenet model without any modifications to
the original model definition.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
import tensorflow as tf

from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.training.python.training import evaluation

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
         'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')

# Model specific paramenters
flags.DEFINE_string(
    'data_dir', '',
    'Directory where input data is stored')

flags.DEFINE_string(
    'model_dir', None,
    'Directory where model output is stored')

flags.DEFINE_integer(
    'num_shards', 8,
    'Number of shards (workers).')

flags.DEFINE_integer(
    'iterations', 100,
    'Number of iterations per TPU training loop.')

flags.DEFINE_integer(
    'train_batch_size', 1024,
    'Global (not per-shard) batch size for training')

flags.DEFINE_integer(
    'eval_total_size', 0,
    'Total batch size for evaluation, use the entire validation set if 0')

flags.DEFINE_integer(
    'eval_batch_size', 1024,
    'Global (not per-shard) batch size for evaluation')

flags.DEFINE_integer(
    'train_steps', 8000000,
    'Number of steps use for training.')

flags.DEFINE_integer(
    'train_steps_per_eval', 2000,
    'Number of training steps to run between evaluations.')

flags.DEFINE_string(
    'mode', 'train_and_eval',
    'Mode to run: train, eval, train_and_eval')

flags.DEFINE_integer(
    'min_eval_interval', 180,
    'Minimum number of seconds between evaluations')

flags.DEFINE_integer(
    'eval_timeout', None,
    'Evaluation timeout: Maximum number of seconds that '
    'may elapse while no new checkpoints are observed')

flags.DEFINE_bool(
    'use_tpu', True,
    'Use TPUs rather than plain CPUs')

flags.DEFINE_boolean(
    'per_host_input_for_training', True,
    'If true, input_fn is invoked per host rather than per shard.')

flags.DEFINE_string(
    'use_data', 'real',
    'One of "fake","real"')

flags.DEFINE_float(
    'learning_rate', 0.165,
    'Learning rate.')

flags.DEFINE_float(
    'depth_multiplier', 1.0,
    'Depth Multiplier on Inception')

flags.DEFINE_integer(
    'num_classes', 1001,
    'Number of classes to distinguish')

flags.DEFINE_integer(
    'width', 224,
    'Width of input image')

flags.DEFINE_integer(
    'height', 224,
    'Height of input image')

flags.DEFINE_bool(
    'transpose_enabled', False,
    'Boolean to enable/disable explicit I/O transpose')

flags.DEFINE_bool(
    'use_fused_batchnorm', True,
    'Enable fused batchrnom')

flags.DEFINE_bool(
    'log_device_placement', False,
    'Boolean to enable/disable log device placement')

flags.DEFINE_integer(
    'save_summary_steps', 100,
    'Number of steps which must have run before showing summaries.')

flags.DEFINE_integer(
    'save_checkpoints_secs', 1000,
    'Interval (in seconds) at which the model data '
    'should be checkpointed. Set to 0 to disable.')

flags.DEFINE_bool(
    'moving_average', True,
    'Whether to enable moving average computation on variables')

flags.DEFINE_bool(
    'use_annotated_bbox', False,
    'If true, use annotated bounding box as input to cropping function, '
    'else use full image size')

flags.DEFINE_bool(
    'use_logits', True,
    'Use logits if true, else use predictions')

flags.DEFINE_bool(
    'display_tensors', False,
    'Whether to dump prediction tensors for comparison')

# Dataset specific paramenters
flags.DEFINE_bool(
    'prefetch_enabled', True,
    'Boolean to enable/disable prefetching')

flags.DEFINE_integer(
    'prefetch_dataset_buffer_size', 8 * 1024 * 1024,
    'Number of bytes in read buffer. 0 means no buffering.')

flags.DEFINE_integer(
    'num_files_infeed', 8,
    'Number of training files to read in parallel.')

flags.DEFINE_integer(
    'num_parallel_calls', 64,
    'Number of elements to process in parallel (by mapper)')

flags.DEFINE_integer(
    'initial_shuffle_buffer_size', 1024,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done before any other operations. '
    'Set to 0 to disable')

flags.DEFINE_integer(
    'followup_shuffle_buffer_size', 1000,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done after prefetching is done. '
    'Set to 0 to disable')

flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

flags.DEFINE_float(
    'batch_norm_decay', None, '')

flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

flags.DEFINE_boolean(
    'preprocessing_fast_mode', True, '')

flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')


flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')


flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')


flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.train_batch_size *
                    FLAGS.num_epochs_per_decay)

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


FLAGS = flags.FLAGS

# Dataset constants
_NUM_TRAIN_IMAGES = 1281167
_NUM_EVAL_IMAGES = 50000

# Random cropping constants
_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

# Constants dictating moving average.
MOVING_AVERAGE_DECAY = 0.995

# Batchnorm moving mean/variance parameters
BATCH_NORM_DECAY = 0.996
BATCH_NORM_EPSILON = 1e-3


class InputPipeline(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The fortmat of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py

  Args:
    is_training: `bool` for whether the input is for training
  """

  def __init__(self, is_training, data_dir):
    self.is_training = is_training
    self.data_dir = data_dir

  def dataset_parser(self, serialized_proto):
    """Parse an Imagenet record from value."""
    keys_to_features = {
      'image/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
        tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label':
        tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text':
        tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/object/bbox/xmin':
        tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
        tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
        tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
        tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
        tf.VarLenFeature(dtype=tf.int64),
    }

    features = tf.parse_single_example(serialized_proto, keys_to_features)

    bbox = None
    if FLAGS.use_annotated_bbox:
      xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
      ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
      xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
      ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

      # Note that we impose an ordering of (y, x) just to make life difficult.
      bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

      # Force the variable number of bounding boxes into the shape
      # [1, num_boxes, coords].
      bbox = tf.expand_dims(bbox, 0)
      bbox = tf.transpose(bbox, [0, 2, 1])

    image = features['image/encoded']
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        FLAGS.preprocessing_name or FLAGS.model_name,
        is_training=self.is_training)
    image = image_preprocessing_fn(image, FLAGS.height, FLAGS.width,
                                   fast_mode=FLAGS.preprocessing_fast_mode)

    label = tf.cast(
        tf.reshape(features['image/class/label'], shape=[]), dtype=tf.int32)

    return image, label

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A (images, labels) tuple of `Tensor`s for a batch of samples.
    """
    batch_size = params['batch_size']

    if FLAGS.use_data == 'real':
      file_pattern = os.path.join(
          self.data_dir, 'train-*' if self.is_training else 'validation-*')
      dataset = tf.data.Dataset.list_files(file_pattern,
                                           shuffle=self.is_training)

      if self.is_training and FLAGS.initial_shuffle_buffer_size:
          dataset = dataset.shuffle(FLAGS.initial_shuffle_buffer_size)

      def prefetch_dataset(filename):
        dataset = tf.data.TFRecordDataset(
            filename, buffer_size=FLAGS.prefetch_dataset_buffer_size)
        return dataset

      dataset = dataset.apply(
          tf.contrib.data.parallel_interleave(
              prefetch_dataset,
              cycle_length=FLAGS.num_files_infeed,
              sloppy=True))

      if self.is_training:
        if FLAGS.followup_shuffle_buffer_size:
          dataset = dataset.apply(
              tf.contrib.data.shuffle_and_repeat(
                  FLAGS.followup_shuffle_buffer_size))
        else:
          dataset = dataset.repeat()

      dataset = dataset.apply(
          tf.contrib.data.map_and_batch(
              self.dataset_parser, batch_size,
              num_parallel_batches=FLAGS.num_parallel_calls,
              drop_remainder=True))

      dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)  # Prefetch overlaps in-feed with training

      images, labels = dataset.make_one_shot_iterator().get_next()
      images.set_shape([batch_size, FLAGS.height, FLAGS.width, 3])
    else:
      images = tf.random_uniform(
          [batch_size, FLAGS.height, FLAGS.width, 3], minval=-1, maxval=1)
      labels = tf.random_uniform(
          [batch_size], minval=0, maxval=999, dtype=tf.int32)

    images = tensor_transform_fn(images, params['output_perm'])
    return images, labels


def tensor_transform_fn(data, perm):
  """Transpose function.

  This function is used to transpose an image tensor on the host and then
  perform an inverse transpose on the TPU. The transpose on the TPU gets
  effectively elided thus voiding any associated computational cost.

  NOTE: Eventually the compiler will be able to detect when this kind of
  operation may prove beneficial and perform these types of transformations
  implicitly, voiding the need for user intervention

  Args:
    data: Tensor to be transposed
    perm: Permutation of the dimensions of a

  Returns:
    Transposed tensor
  """
  if FLAGS.transpose_enabled:
    return tf.transpose(data, perm)
  return data


def model_fn(features, labels, mode, params):
  """Mobilenet v1 model using Estimator API."""

  network_fn = nets_factory.get_network_fn(
      FLAGS.model_name,
      num_classes=FLAGS.num_classes,
      weight_decay=FLAGS.weight_decay,
      is_training=True,
      batch_norm_decay=FLAGS.batch_norm_decay)

  training_active = (mode == tf.estimator.ModeKeys.TRAIN)
  eval_active = (mode == tf.estimator.ModeKeys.EVAL)

  features = tensor_transform_fn(features, params['input_perm'])

  logits, end_points = network_fn(features)
  predictions = {
    'classes': tf.argmax(input=logits, axis=1),
    'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  if mode == tf.estimator.ModeKeys.EVAL and FLAGS.display_tensors and (
      not FLAGS.use_tpu):
    with tf.control_dependencies([
      tf.Print(
          predictions['classes'], [predictions['classes']],
          summarize=FLAGS.eval_batch_size,
          message='prediction: ')
    ]):
      labels = tf.Print(
          labels, [labels], summarize=FLAGS.eval_batch_size, message='label: ')

  one_hot_labels = tf.one_hot(labels, FLAGS.num_classes, dtype=tf.int32)

  tf.losses.softmax_cross_entropy(
      onehot_labels=one_hot_labels,
      logits=logits,
      weights=1.0,
      label_smoothing=FLAGS.label_smoothing)
  loss = tf.losses.get_total_loss(add_regularization_losses=True)

  train_op = None
  if training_active:
    global_step = tf.train.get_or_create_global_step()
    learning_rate = _configure_learning_rate(_NUM_TRAIN_IMAGES, global_step)
    optimizer = _configure_optimizer(learning_rate)

    if FLAGS.use_tpu:
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=global_step)
    if FLAGS.moving_average:
      ema = tf.train.ExponentialMovingAverage(
          decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
      variables_to_average = (tf.trainable_variables() +
                              tf.moving_average_variables())
      with tf.control_dependencies([train_op]), tf.name_scope('moving_average'):
        train_op = ema.apply(variables_to_average)

  eval_metrics = None
  if eval_active:
    def metric_fn(labels, predictions):
      accuracy = tf.metrics.accuracy(labels, tf.argmax(
          input=predictions, axis=1))
      return {'accuracy': accuracy}

    if FLAGS.use_logits:
      eval_predictions = logits
    else:
      eval_predictions = end_points['Predictions']

    eval_metrics = (metric_fn, [labels, eval_predictions])

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode, loss=loss, train_op=train_op, eval_metrics=eval_metrics)


class LoadEMAHook(tf.train.SessionRunHook):
  """Hook to load EMA into their corresponding variables."""

  def __init__(self, model_dir):
    super(LoadEMAHook, self).__init__()
    self._model_dir = model_dir

  def begin(self):
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = ema.variables_to_restore()
    self._load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
        tf.train.latest_checkpoint(self._model_dir), variables_to_restore)

  def after_create_session(self, sess, coord):
    tf.logging.info('Reloading EMA...')
    self._load_ema(sess)


def main(unused_argv):
  del unused_argv  # Unused

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  batch_size_per_shard = FLAGS.train_batch_size // FLAGS.num_shards
  params = {
    'input_perm': [0, 1, 2, 3],
    'output_perm': [0, 1, 2, 3],
  }

  batch_axis = 0
  if FLAGS.transpose_enabled:
    if batch_size_per_shard >= 64:
      params['input_perm'] = [3, 0, 1, 2]
      params['output_perm'] = [1, 2, 3, 0]
      batch_axis = 3
    else:
      params['input_perm'] = [2, 0, 1, 3]
      params['output_perm'] = [1, 2, 0, 3]
      batch_axis = 2

  if FLAGS.eval_total_size > 0:
    eval_size = FLAGS.eval_total_size
  else:
    eval_size = _NUM_EVAL_IMAGES
  eval_steps = eval_size // FLAGS.eval_batch_size

  iterations = (eval_steps if FLAGS.mode == 'eval' else
                FLAGS.iterations)

  eval_batch_size = (None if FLAGS.mode == 'train' else
                     FLAGS.eval_batch_size)

  per_host_input_for_training = (FLAGS.num_shards <= 8 if
                                 FLAGS.mode == 'train' else True)

  run_config = tpu_config.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_summary_steps=FLAGS.save_summary_steps,
      session_config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement),
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=iterations,
          num_shards=FLAGS.num_shards,
          per_host_input_for_training=per_host_input_for_training))

  inception_classifier = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      params=params,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=eval_batch_size,
      batch_axis=(batch_axis, 0))

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  imagenet_train = InputPipeline(
      is_training=True,
      data_dir=FLAGS.data_dir)
  imagenet_eval = InputPipeline(
      is_training=False,
      data_dir=FLAGS.data_dir)

  if FLAGS.moving_average:
    eval_hooks = [LoadEMAHook(FLAGS.model_dir)]
  else:
    eval_hooks = []

  if FLAGS.mode == 'eval':
    def terminate_eval():
      tf.logging.info('%d seconds without new checkpoints have elapsed '
                      '... terminating eval' % FLAGS.eval_timeout)
      return True

    def get_next_checkpoint():
      return evaluation.checkpoints_iterator(
          FLAGS.model_dir,
          min_interval_secs=FLAGS.min_eval_interval,
          timeout=FLAGS.eval_timeout,
          timeout_fn=terminate_eval)

    for checkpoint in get_next_checkpoint():
      tf.logging.info('Starting to evaluate.')
      try:
        eval_results = inception_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            hooks=eval_hooks,
            checkpoint_path=checkpoint)
        tf.logging.info('Evaluation results: %s' % eval_results)
      except tf.errors.NotFoundError:
        # skip checkpoint if it gets deleted prior to evaluation
        tf.logging.info('Checkpoint %s no longer exists ... skipping')

  elif FLAGS.mode == 'train_and_eval':
    for cycle in range(FLAGS.train_steps // FLAGS.train_steps_per_eval):
      tf.logging.info('Starting training cycle %d.' % cycle)
      inception_classifier.train(
          input_fn=imagenet_train.input_fn, steps=FLAGS.train_steps_per_eval)

      tf.logging.info('Starting evaluation cycle %d .' % cycle)
      eval_results = inception_classifier.evaluate(
          input_fn=imagenet_eval.input_fn, steps=eval_steps, hooks=eval_hooks)
      tf.logging.info('Evaluation results: %s' % eval_results)

  else:
    tf.logging.info('Starting training ...')
    inception_classifier.train(
        input_fn=imagenet_train.input_fn, steps=FLAGS.train_steps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
