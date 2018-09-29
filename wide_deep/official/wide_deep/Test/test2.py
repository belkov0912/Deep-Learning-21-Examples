import tensorflow as tf

def parse_string(x):
    x = tf.strings.strip(x)
    x = tf.strings.split([x], ' ')
    l = x.dense_shape[0]
    x = tf.reshape(x, l)[0:200]
    return x

def test():
    sess = tf.Session()

    a = tf.constant("46 57 aa")

    x = tf.strings.strip(a)
    x = tf.strings.split([x], ' ')
    # y = tf.sparse_to_dense(x.indices, [3], x.values, default_value="")
    y = tf.strings.join([x.values])
    # y = 0
    # l = x.dense_shape[3]
    # x = tf.reshape(xl)

    l, m = sess.run([x, y])
    print(l, m)

    # b = tf.string_split([a], delimiter="|", skip_empty=True)
    # d = tf.string_to_number(b, out_type=tf.int32)
    # print(sess.run(d))

test()


# def sparse_tensor_merge(indices, values, shape):
#   """Creates a SparseTensor from batched indices, values, and shapes.
#
#   Args:
#     indices: A [batch_size, N, D] integer Tensor.
#     values: A [batch_size, N] Tensor of any dtype.
#     shape: A [batch_size, D] Integer Tensor.
#   Returns:
#     A SparseTensor of dimension D + 1 with batch_size as its first dimension.
#   """
#   merged_shape = tf.reduce_max(shape, axis=0)
#   batch_size, elements, shape_dim = tf.unstack(tf.shape(indices))
#   index_range_tiled = tf.tile(tf.range(batch_size)[..., None],
#                               tf.stack([1, elements]))[..., None]
#   merged_indices = tf.reshape(
#       tf.concat([tf.cast(index_range_tiled, tf.int64), indices], axis=2),
#       [-1, 1 + tf.size(merged_shape)])
#   merged_values = tf.reshape(values, [-1])
#   return tf.SparseTensor(
#       merged_indices, merged_values,
#       tf.concat([[tf.cast(batch_size, tf.int64)], merged_shape], axis=0))
#
# batch_indices = tf.constant(
#     [[[0, 0], [0, 1]],
#      [[0, 0], [1, 1]]], dtype=tf.int64)
# batch_values = tf.constant(
#     [[0.1, 0.2],
#      [0.3, 0.4]])
# batch_shapes = tf.constant(
#     [[2, 2],
#      [3, 2]], dtype=tf.int64)
#
# merged = sparse_tensor_merge(batch_indices, batch_values, batch_shapes)
#
# with tf.Session():
#   print(merged.eval())