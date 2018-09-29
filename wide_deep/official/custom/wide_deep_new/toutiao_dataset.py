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
"""Download and clean the Census Income Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# pylint: disable=wrong-import-order
from absl import app as absl_app
from absl import flags
from six.moves import urllib
import tensorflow as tf
# pylint: enable=wrong-import-order

from official.utils.flags import core as flags_core
from tensorflow.python.framework import dtypes
import pandas as pd
import math

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'train.data_sample'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'test.test_sample'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)


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

_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

# df = pd.read_csv(flags.FLAGS.data_dir + "/" + TRAINING_FILE, names=_CSV_COLUMNS)
df = pd.read_csv("/Users/jiananliu/work/machinelearn/dl/wide_deep/data" + "/" + TRAINING_FILE, names=_CSV_COLUMNS)
df_describe = df.describe()
df_mean = df_describe.loc["mean"]
df_std = df_describe.loc["std"]
df_count = df_describe.loc["count"]
df_min = df_describe.loc["min"]
df_max = df_describe.loc["max"]

def normalize(val, col_name):
    #0.5*math.log((val - df_min[col_name]) / (df_mean[col_name] - df_min[col_name]) + 1, 2)
    #return (val - df_mean[col_name]) / df_std[col_name]
    return (val - df_mean[col_name])/max(df_std[col_name], 1e-10)

def _download_and_clean_file(filename, url):
  """Downloads data from url, and makes changes to match the CSV format."""
  temp_file, _ = urllib.request.urlretrieve(url)
  with tf.gfile.Open(temp_file, 'r') as temp_eval_file:
    with tf.gfile.Open(filename, 'w') as eval_file:
      for line in temp_eval_file:
        line = line.strip()
        line = line.replace(', ', ',')
        if not line or ',' not in line:
          continue
        if line[-1] == '.':
          line = line[:-1]
        line += '\n'
        eval_file.write(line)
  tf.gfile.Remove(temp_file)


def download(data_dir):
  """Download census data if it is not already present."""
  tf.gfile.MakeDirs(data_dir)

  training_file_path = os.path.join(data_dir, TRAINING_FILE)
  if not tf.gfile.Exists(training_file_path):
    _download_and_clean_file(training_file_path, TRAINING_URL)

  eval_file_path = os.path.join(data_dir, EVAL_FILE)
  if not tf.gfile.Exists(eval_file_path):
    _download_and_clean_file(eval_file_path, EVAL_URL)



def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous columns
    # gender = tf.feature_column.numeric_column("gender", shape=(1,), default_value=-1, dtype=dtypes.int8)
    # age = tf.feature_column.numeric_column("age", shape=(1,),default_value=-1, dtype=dtypes.int8)
    createtime = tf.feature_column.numeric_column('createtime', dtype=dtypes.int64, normalizer_fn=lambda x:normalize(x, 'createtime'))
    addtime = tf.feature_column.numeric_column('addtime', dtype=dtypes.int64, normalizer_fn=lambda x:normalize(x, 'addtime'))
    clickmax = tf.feature_column.numeric_column('clickmax', normalizer_fn=lambda x:normalize(x, 'clickmax'))
    clickcount = tf.feature_column.numeric_column('clickcount', normalizer_fn=lambda x:normalize(x, 'clickcount'))
    showcount = tf.feature_column.numeric_column('showcount', normalizer_fn=lambda x:normalize(x, 'showcount'))
    likemax = tf.feature_column.numeric_column('likemax', normalizer_fn=lambda x:normalize(x, 'likemax'))
    likecount = tf.feature_column.numeric_column('likecount', normalizer_fn=lambda x:normalize(x, 'likecount'))
    dislikecount = tf.feature_column.numeric_column('dislikecount', normalizer_fn=lambda x:normalize(x, 'dislikecount'))
    showgoodpercent = tf.feature_column.numeric_column('showgoodpercent', normalizer_fn=lambda x:normalize(x, 'showgoodpercent'))
    allclicknum = tf.feature_column.numeric_column('allclicknum', normalizer_fn=lambda x:normalize(x, 'allclicknum'))
    allshownum = tf.feature_column.numeric_column('allshownum', normalizer_fn=lambda x:normalize(x, 'allshownum'))
    sharenum = tf.feature_column.numeric_column('sharenum', normalizer_fn=lambda x:normalize(x, 'sharenum'))
    favoritenum = tf.feature_column.numeric_column('favoritenum', normalizer_fn=lambda x:normalize(x, 'favoritenum'))
    cateid = tf.feature_column.numeric_column('cateid')
    # topicid = tf.feature_column.categorical_column_with_identity('topicid', num_buckets=1000)
    topicid = tf.feature_column.categorical_column_with_vocabulary_list(
        'topicid', [str(i) for i in range(0, 100)], dtype=tf.string, default_value=-1
    )
    keywordid = tf.feature_column.numeric_column('keywordid')
    cf_score = tf.feature_column.numeric_column('cf_score', normalizer_fn=lambda x:normalize(x, 'cf_score'))
    topic_score = tf.feature_column.numeric_column('topic_score', normalizer_fn=lambda x:normalize(x, 'topic_score'))
    keyword_score = tf.feature_column.numeric_column('keyword_score', normalizer_fn=lambda x:normalize(x, 'keyword_score'))
    topicctr_score = tf.feature_column.numeric_column('topicctr_score', normalizer_fn=lambda x:normalize(x, 'topicctr_score'))
    topic_score_short = tf.feature_column.numeric_column('topic_score_short', normalizer_fn=lambda x:normalize(x, 'topic_score_short'))
    topicctr_score_short = tf.feature_column.numeric_column('topicctr_score_short', normalizer_fn=lambda x:normalize(x, 'topicctr_score_short'))
    dt_score = tf.feature_column.numeric_column('dt_score', normalizer_fn=lambda x:normalize(x, 'dt_score'))
    toutiaotaste_score = tf.feature_column.numeric_column('toutiaotaste_score', normalizer_fn=lambda x:normalize(x, 'toutiaotaste_score'))
    apptaste_score = tf.feature_column.numeric_column('apptaste_score', normalizer_fn=lambda x:normalize(x, 'apptaste_score'))

    gender_cate = tf.feature_column.categorical_column_with_vocabulary_list(key="gender",
                                                                            vocabulary_list=[-1, 0, 1],
                                                                            dtype=tf.int8,
                                                                            default_value=-1,
                                                                            num_oov_buckets=3)
    age_cate = tf.feature_column.categorical_column_with_vocabulary_list(key="age",
                                                                         vocabulary_list=[-1, 0, 1, 2, 3, 4, 5, 6, ],
                                                                         dtype=tf.int8,
                                                                         default_value=-1,
                                                                         num_oov_buckets=7)
    # Wide columns and deep columns.
    base_columns = [
        gender_cate, age_cate, clickmax, clickcount, showcount, likemax, likecount, dislikecount, showgoodpercent,
        allclicknum, allshownum,
        sharenum, favoritenum, cf_score, topic_score, keyword_score, topicctr_score, topic_score_short,
        topicctr_score_short,
        dt_score, toutiaotaste_score, apptaste_score,
        topicid
    ]

    wide_columns = base_columns  # + crossed_columns

    deep_columns = [
        clickmax, clickcount, showcount, likemax, likecount, dislikecount, showgoodpercent,
        allclicknum, allshownum,
        sharenum, favoritenum, cf_score, topic_score, keyword_score, topicctr_score, topic_score_short,
        topicctr_score_short,
        dt_score, toutiaotaste_score, apptaste_score,
        tf.feature_column.indicator_column(gender_cate),
        tf.feature_column.indicator_column(age_cate)
    ]

    return wide_columns, deep_columns

def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run census_dataset.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    tf.logging.info('Parsing {}'.format(data_file))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    topics = tf.string_split([features.get("topicid")], "|")
    # tsv = tf.string_to_number(topics.values, out_type=dtypes.int32)
    features["topicid"] = topics
    labels = features.pop('label')
    classes = tf.equal(labels, 1.0)  # binary classification
    return features, classes

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


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", default="/tmp/census_data/",
      help=flags_core.help_wrap(
          "Directory to download and extract data."))


def main(_):
  download(flags.FLAGS.data_dir)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_data_download_flags()
  absl_app.run(main)
