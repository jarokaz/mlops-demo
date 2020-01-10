
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma
from tensorflow_transform.tf_metadata import schema_utils

DENSE_FLOAT_FEATURE_KEYS = [
    "timedelta", "n_tokens_title", "n_tokens_content",
    "n_unique_tokens", "n_non_stop_words", "n_non_stop_unique_tokens",
    "n_hrefs", "n_self_hrefs", "n_imgs", "n_videos", "average_token_length",
    "n_keywords", "kw_min_min", "kw_max_min", "kw_avg_min", "kw_min_max",
    "kw_max_max", "kw_avg_max", "kw_min_avg", "kw_max_avg", "kw_avg_avg",
    "self_reference_min_shares", "self_reference_max_shares",
    "self_reference_avg_shares", "is_weekend", "global_subjectivity",
    "global_sentiment_polarity", "global_rate_positive_words",
    "global_rate_negative_words", "rate_positive_words", "rate_negative_words",
    "avg_positive_polarity", "min_positive_polarity", "max_positive_polarity",
    "avg_negative_polarity", "min_negative_polarity", "max_negative_polarity",
    "title_subjectivity", "title_sentiment_polarity", "abs_title_subjectivity",
    "abs_title_sentiment_polarity"]

VOCAB_FEATURE_KEYS = ["data_channel"]

BUCKET_FEATURE_KEYS = ["LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04"]

CATEGORICAL_FEATURE_KEYS = ["weekday"]

# Categorical features are assumed to each have a maximum value in the dataset.
MAX_CATEGORICAL_FEATURE_VALUES = [6]

#UNUSED: date, slug

LABEL_KEY = "n_shares_percentile"
VOCAB_SIZE = 10
OOV_SIZE = 5
FEATURE_BUCKET_COUNT = 10

def transformed_name(key):
  return key + '_xf'

def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in DENSE_FLOAT_FEATURE_KEYS:
    # Preserve this feature as a dense float, setting nan's to the mean.
    outputs[transformed_name(key)] = tft.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(
        _fill_in_missing(inputs[key]),
        top_k=VOCAB_SIZE,
        num_oov_buckets=OOV_SIZE)

  for key in BUCKET_FEATURE_KEYS:
    outputs[transformed_name(key)] = tft.bucketize(
        _fill_in_missing(inputs[key]), FEATURE_BUCKET_COUNT,
        always_return_num_quantiles=False)

  for key in CATEGORICAL_FEATURE_KEYS:
    outputs[transformed_name(key)] = _fill_in_missing(inputs[key])

  # How popular is this article?
  outputs[transformed_name(LABEL_KEY)] = _fill_in_missing(inputs[LABEL_KEY])

  return outputs

def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.

  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.

  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)

def transformed_names(keys):
  return [transformed_name(key) for key in keys]


# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')


def _build_estimator(config, hidden_units=None, warm_start_from=None):
  """Build an estimator for predicting the popularity of online news articles

  Args:
    config: tf.estimator.RunConfig defining the runtime environment for the
      estimator (including model_dir).
    hidden_units: [int], the layer sizes of the DNN (input layer first)
    warm_start_from: Optional directory to warm start from.

  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """
  real_valued_columns = [
      tf.feature_column.numeric_column(key, shape=())
      for key in transformed_names(DENSE_FLOAT_FEATURE_KEYS)
  ]
  categorical_columns = [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=VOCAB_SIZE + OOV_SIZE, default_value=0)
      for key in transformed_names(VOCAB_FEATURE_KEYS)
  ]
  categorical_columns += [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=FEATURE_BUCKET_COUNT, default_value=0)
      for key in transformed_names(BUCKET_FEATURE_KEYS)
  ]
  categorical_columns += [
      tf.feature_column.categorical_column_with_identity(
          key,
          num_buckets=num_buckets,
          default_value=0) for key, num_buckets in zip(
              transformed_names(CATEGORICAL_FEATURE_KEYS),
              MAX_CATEGORICAL_FEATURE_VALUES)
  ]
  return tf.estimator.DNNLinearCombinedRegressor(
      config=config,
      linear_feature_columns=categorical_columns,
      dnn_feature_columns=real_valued_columns,
      dnn_hidden_units=hidden_units or [100, 70, 50, 25],
      warm_start_from=warm_start_from)


def _example_serving_receiver_fn(tf_transform_output, schema):
  """Build the serving in inputs.

  Args:
    tf_transform_output: A TFTransformOutput.
    schema: the schema of the input data.

  Returns:
    Tensorflow graph which parses examples, applying tf-transform to them.
  """
  raw_feature_spec = _get_raw_feature_spec(schema)
  raw_feature_spec.pop(LABEL_KEY)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()

  transformed_features = tf_transform_output.transform_raw_features(
      serving_input_receiver.features)

  return tf.estimator.export.ServingInputReceiver(
      transformed_features, serving_input_receiver.receiver_tensors)


def _eval_input_receiver_fn(tf_transform_output, schema):
  """Build everything needed for the tf-model-analysis to run the model.

  Args:
    tf_transform_output: A TFTransformOutput.
    schema: the schema of the input data.

  Returns:
    EvalInputReceiver function, which contains:
      - Tensorflow graph which parses raw untransformed features, applies the
        tf-transform preprocessing operators.
      - Set of raw, untransformed features.
      - Label against which predictions will be compared.
  """
  # Notice that the inputs are raw features, not transformed features here.
  raw_feature_spec = _get_raw_feature_spec(schema)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()

  features = serving_input_receiver.features.copy()
  transformed_features = tf_transform_output.transform_raw_features(features)
  
  # NOTE: Model is driven by transformed features (since training works on the
  # materialized output of TFT, but slicing will happen on raw features.
  features.update(transformed_features)

  return tfma.export.EvalInputReceiver(
      features=features,
      receiver_tensors=serving_input_receiver.receiver_tensors,
      labels=transformed_features[transformed_name(LABEL_KEY)])


def _input_fn(filenames, tf_transform_output, batch_size=200):
  """Generates features and labels for training or evaluation.

  Args:
    filenames: [str] list of CSV files to read data from.
    tf_transform_output: A TFTransformOutput.
    batch_size: int First dimension size of the Tensors returned by input_fn

  Returns:
    A (features, indices) tuple where features is a dictionary of
      Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

  transformed_features = dataset.make_one_shot_iterator().get_next()
  # We pop the label because we do not want to use it as a feature while we're
  # training.
  return transformed_features, transformed_features.pop(
      transformed_name(LABEL_KEY))


# TFX will call this function
def trainer_fn(hparams, schema):
  """Build the estimator using the high level API.
  Args:
    hparams: Holds hyperparameters used to train the model as name/value pairs.
    schema: Holds the schema of the training examples.
  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """
  # Number of nodes in the first layer of the DNN
  first_dnn_layer_size = 100
  num_dnn_layers = 4
  dnn_decay_factor = 0.7

  train_batch_size = 40
  eval_batch_size = 40

  tf_transform_output = tft.TFTransformOutput(hparams.transform_output)

  train_input_fn = lambda: _input_fn(
      hparams.train_files,
      tf_transform_output,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(
      hparams.eval_files,
      tf_transform_output,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(
      train_input_fn,
      max_steps=hparams.train_steps)

  serving_receiver_fn = lambda: _example_serving_receiver_fn(
      tf_transform_output, schema)

  exporter = tf.estimator.FinalExporter('online-news', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=hparams.eval_steps,
      exporters=[exporter],
      name='online-news-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=hparams.serving_model_dir)

  estimator = _build_estimator(
      # Construct layers sizes with exponetial decay
      hidden_units=[
          max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
          for i in range(num_dnn_layers)
      ],
      config=run_config,
      warm_start_from=hparams.warm_start_from)

  # Create an input receiver for TFMA processing
  receiver_fn = lambda: _eval_input_receiver_fn(
      tf_transform_output, schema)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': receiver_fn
  }
