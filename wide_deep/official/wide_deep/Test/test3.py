import tensorflow as tf
from tensorflow.python.estimator.inputs import numpy_io
import numpy as np
import collections
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow import feature_column

from tensorflow.python.feature_column.feature_column import _LazyBuilder

def test_numeric():
    # price = {'price': [[1.], [2.], [3.], [4.]]}  # 4行样本
    # price = {'price': ['123|45|6']}  # 4行样本
    # price = {'price': ['123', '45', '6']}  # 4行样本
    price = {'price': [['123'], ['45'], ['6']]}  # 4行样本
    builder = _LazyBuilder(price)

    def transform_fn(x):
        aa = tf.Print(x, [])
        with tf.Session() as sess:
            print(sess.run(aa))
        print(x.shape)
        # aa = tf.string_split(tf.reshape(x, [3,]), '|')
        return tf.string_to_number(x) + 2

    # price_column = feature_column.numeric_column('price', normalizer_fn=transform_fn)
    price_column = feature_column.categorical_column_with_vocabulary_list(
        'price', ['123', '45', '6'], dtype=tf.string, default_value=-1
    )

    # price_transformed_tensor = price_column._get_dense_tensor(builder)
    price_transformed_tensor = price_column._get_sparse_tensors(builder)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        session.run(tf.tables_initializer())
        print(session.run([price_transformed_tensor.id_tensor]))

    # 使用input_layer

    # price_transformed_tensor = feature_column.input_layer(price, [price_column])
    #
    # with tf.Session() as session:
    #     print('use input_layer' + '_' * 40)
    #     print(session.run([price_transformed_tensor]))

# test_numeric()

def test_bucketized_column():
    # input = '5|15|25|35|'


    price = {'price': [[5.], [15.], [25.], [35.]]}  # 4行样本

    price_column = feature_column.numeric_column('price')
    bucket_price = feature_column.bucketized_column(price_column, [0, 10, 20, 30, 40])

    price_bucket_tensor = feature_column.input_layer(price, [bucket_price])

    with tf.Session() as session:
        print(session.run([price_bucket_tensor]))

# test_bucketized_column()


# import tensorflow as tf
# import tempfile
# import urllib
# train_file = tempfile.NamedTemporaryFile()
# test_file = tempfile.NamedTemporaryFile()
# urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',train_file.name)
# urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',test_file.name)
#
# CATEGORICAL_COLUMNS = ['workclass','education','marital_status','occupation','relationship','race','gender','native_country']
# CONTINUOUS_COLUMNS = ['age','education_num','capital_gain','capital_loss','hours_per_week']
# def input_fn(df):
#     #Creates a dictionary mapping from each continuous feature column name(k) to
#     #the values of that column stored in a constant Tensor.
#     continuous_cols = {k:tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
#     #Creates a dictionary mapping from each categorical feature column name(k) to the values of that column stored in a tf.SparseTensor.
#     categorical_cols = {k:tf.SparseTensor(indices[[i,0] for i in range(df[k].size)], values=df[k].values,dense_shape=[df[k].size,1]) for k in CATEGORICAL_COLUMNS}
#     #把每一个categorical特征都挑出来，每个特征构成1列32561行(普查的人数)的矩阵，由于k是一直在变的，所以最终的categorical_cols.items有八个，皆是categorical特征。
#     #Merges the two dictionaries into one.
#     feature_cols = dict(continuous_cols.items()|categorical_cols.items())
#     #Converts the label column into a constant Tensor.
#     label = tf.constant(df[LABEL_COLUMN].values)
#     #Returns the feature columns and the label.
#     return feature_cols,label
#
# def train_input_fn():
#     return input_fn(df_train)
# def eval_input_fn():
#     return input_fn(df_test)

def test_categorical_column_with_vocabulary_list():

    color_data = {'color': [['R', 'R'], ['G', 'R'], ['B', 'G'], ['A', 'A']]}  # 4行样本

    builder = _LazyBuilder(color_data)

    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )

    color_column_tensor = color_column._get_sparse_tensors(builder)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        session.run(tf.tables_initializer())

        print(session.run([color_column_tensor.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_identy = feature_column.indicator_column(color_column)

    color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        session.run(tf.tables_initializer())

        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))

