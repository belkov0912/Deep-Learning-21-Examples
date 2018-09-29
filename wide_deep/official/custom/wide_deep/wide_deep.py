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
"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys
import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.arg_parsers import parsers
from official.utils.logging import hooks_helper

from tensorflow.python.framework import dtypes
_CSV_COLUMNS = [
    'gender', 'age', 'createtime', 'addtime', 'clickmax',
    'clickcount', 'showcount', 'likemax', 'likecount', 'dislikecount',
    'showgoodpercent', 'allclicknum', 'allshownum', 'sharenum', 'favoritenum',
    'cateid', 'topicid', 'keywordid', 'cf_score', 'topic_score',
    'keyword_score','topicctr_score', 'topic_score_short', 'topicctr_score_short', 'dt_score', 
    'toutiaotaste_score', 'apptaste_score','label'
]

_CSV_COLUMN_DEFAULTS = [[-1], [-1], [1278917023000], [1278917023000], [0.0],
                        [0.0], [0.0], [0.0], [0.0], [0.0],
                        [0.0], [0.0], [0.0], [0.0], [0.0],
                        [''], [''], [''], [0.0], [0.0],
                        [0.0], [0.0], [0.0], [0.0], [0.0],
                        [0.0], [0.0], [0.0]]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  #gender = tf.feature_column.numeric_column("gender", shape=(1,), default_value=-1, dtype=dtypes.int8)
  #age = tf.feature_column.numeric_column("age", shape=(1,),default_value=-1, dtype=dtypes.int8)
  createtime = tf.feature_column.numeric_column('createtime', dtype=dtypes.int64)
  addtime = tf.feature_column.numeric_column('addtime', dtype=dtypes.int64)
  clickmax = tf.feature_column.numeric_column('clickmax')
  clickcount = tf.feature_column.numeric_column('clickcount')
  showcount = tf.feature_column.numeric_column('showcount')
  likemax = tf.feature_column.numeric_column('likemax')
  likecount = tf.feature_column.numeric_column('likecount')
  dislikecount = tf.feature_column.numeric_column('dislikecount')
  showgoodpercent = tf.feature_column.numeric_column('showgoodpercent')
  allclicknum = tf.feature_column.numeric_column('allclicknum')
  allshownum = tf.feature_column.numeric_column('allshownum')
  sharenum = tf.feature_column.numeric_column('sharenum')
  favoritenum = tf.feature_column.numeric_column('favoritenum')
  cateid = tf.feature_column.numeric_column('cateid')
  topicid = tf.feature_column.numeric_column('topicid')
  keywordid = tf.feature_column.numeric_column('keywordid')
  cf_score = tf.feature_column.numeric_column('cf_score')
  topic_score = tf.feature_column.numeric_column('topic_score')
  keyword_score = tf.feature_column.numeric_column('keyword_score')
  topicctr_score = tf.feature_column.numeric_column('topicctr_score')
  topic_score_short = tf.feature_column.numeric_column('topic_score_short')
  topicctr_score_short = tf.feature_column.numeric_column('topicctr_score_short')
  dt_score = tf.feature_column.numeric_column('dt_score')
  toutiaotaste_score = tf.feature_column.numeric_column('toutiaotaste_score')
  apptaste_score = tf.feature_column.numeric_column('apptaste_score')

  gender_cate = tf.feature_column.categorical_column_with_vocabulary_list(key = "gender",
                                                                            vocabulary_list=[-1, 0 , 1],
                                                                            dtype=tf.int8,
                                                                            default_value=-1,
                                                                            num_oov_buckets=3)
  age_cate = tf.feature_column.categorical_column_with_vocabulary_list(key = "age",
                                                                            vocabulary_list=[-1, 0 , 1, 2, 3, 4, 5, 6,],
                                                                            dtype=tf.int8,
                                                                            default_value=-1,
                                                                            num_oov_buckets=7)
# Wide columns and deep columns.
  base_columns = [
      gender_cate, age_cate, clickmax, clickcount, showcount, likemax, likecount, dislikecount, showgoodpercent, allclicknum, allshownum,
      sharenum, favoritenum, cf_score, topic_score, keyword_score, topicctr_score, topic_score_short, topicctr_score_short,
      dt_score, toutiaotaste_score, apptaste_score
  ]

  wide_columns = base_columns #+ crossed_columns

  deep_columns = [
      clickmax, clickcount, showcount, likemax, likecount, dislikecount, showgoodpercent, allclicknum, allshownum,
      sharenum, favoritenum, cf_score, topic_score, keyword_score, topicctr_score, topic_score_short, topicctr_score_short,
      dt_score, toutiaotaste_score, apptaste_score,
      tf.feature_column.indicator_column(gender_cate),
      tf.feature_column.indicator_column(age_cate)
  ]

  return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)




def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run data_download.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_text(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('label')
    return features, tf.equal(labels, 1.0)


  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('label')
    print(features)
    return features, tf.equal(labels, 1.0)

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset


def main(_):
  # Clean up the model directory if present
  shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
  model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

  train_file = os.path.join(FLAGS.data_dir, 'train.data')
  test_file = os.path.join(FLAGS.data_dir, 'test.test')

  # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
  def train_input_fn():
    return input_fn(train_file, FLAGS.epochs_between_evals, True, FLAGS.batch_size)

  def eval_input_fn():
    return input_fn(test_file, 1, False, FLAGS.batch_size)

  train_hooks = hooks_helper.get_train_hooks(
      FLAGS.hooks, batch_size=FLAGS.batch_size,
      tensors_to_log={'average_loss': 'head/truediv',
                      'loss': 'head/weighted_loss/Sum'})

  # Train and evaluate the model every `FLAGS.epochs_between_evals` epochs.
  for n in range(FLAGS.train_epochs // FLAGS.epochs_between_evals):
    model.train(input_fn=train_input_fn, hooks=train_hooks)
    results = model.evaluate(input_fn=eval_input_fn)

    # Display evaluation metrics
    print('Results at epoch', (n + 1) * FLAGS.epochs_between_evals)
    print('-' * 60)

    for key in sorted(results):
      print('%s: %s' % (key, results[key]))


class WideDeepArgParser(argparse.ArgumentParser):
  """Argument parser for running the wide deep model."""

  def __init__(self):
    super(WideDeepArgParser, self).__init__(parents=[parsers.BaseParser()])
    self.add_argument(
        '--model_type', '-mt', type=str, default='wide_deep',
        choices=['wide', 'deep', 'wide_deep'],
        help='[default %(default)s] Valid model types: wide, deep, wide_deep.',
        metavar='<MT>')
    self.set_defaults(
        data_dir='/tmp/census_data',
        model_dir='/tmp/census_model',
        train_epochs=40,
        epochs_between_evals=2,
        batch_size=40)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = WideDeepArgParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
