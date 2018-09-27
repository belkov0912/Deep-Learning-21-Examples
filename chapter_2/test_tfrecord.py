# 将原始图片转换成需要的大小，并将其保存
#https://blog.csdn.net/ywx1832990/article/details/78609323
# ========================================================================================
import os
import tensorflow as tf
from PIL import Image
import random

# 原始图片的存储位置

orig_picture = '/Users/jiananliu/work/machinelearn/dl/Deep-Learning-21-Examples/book_data/chapter_2/58fang_data/'
file_name = "58fang_train_%d.tfrecords"

# orig_picture = '/Users/jiananliu/work/machinelearn/dl/Deep-Learning-21-Examples/book_data/chapter_2'

# 生成图片的存储位置
gen_picture = '/Users/jiananliu/work/machinelearn/dl/Deep-Learning-21-Examples/chapter_2/cifar10_data/cifar-10-raw-pic'

# 需要的识别类型
# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classes = ['balcony',  'bathroom',  'bedroom',  'bookroom',  'cloakroom',  'dining',  'kitchen',  'living']

# 样本总数
num_samples = 120
heigth = 128
width = 128
channel = 3

# 制作TFRecords数据
def create_record():
    writer = tf.python_io.TFRecordWriter(file_name)
    for index, name in enumerate(classes):
        class_path = orig_picture + "/" + name + "/"
        num = 1000000
        i = 0
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img2 = Image.open(img_path)
            img = img2.resize((heigth, width))  # 设置需要转换的图片大小
            img_raw = img.tobytes()  # 将图片转化为原生bytes
            l = len(img_raw)
            if l != heigth * width * channel:
                continue
            print(index, l)
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
            if i >= num:
                break
            i += 1
    writer.close()


# =======================================================================================
# def read_and_decode(filename):
#     # 创建文件队列,不限读取的数量
#     filename_queue = tf.train.string_input_producer([filename])
#     # create a reader from file queue
#     reader = tf.TFRecordReader()
#     # reader从文件队列中读入一个序列化的样本
#     _, serialized_example = reader.read(filename_queue)
#     # get feature from serialized example
#     # 解析符号化的样本
#     features = tf.parse_single_example(
#         serialized_example,
#         features={
#             'label': tf.FixedLenFeature([], tf.int64),
#             'img_raw': tf.FixedLenFeature([], tf.string)
#         })
#     label = features['label']
#     img = features['img_raw']
#     print(img)
#     img = tf.decode_raw(img, tf.uint8)
#     img = tf.reshape(img, [heigth, width, channel])
#     # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
#     label = tf.cast(label, tf.int32)
#     return img, label

def read_and_decode(filename):

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3

    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })
    label = features['label']
    label = tf.reshape(label, [1])
    label = tf.cast(label, tf.int32)
    result.label = label
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)

    img = tf.reshape(img, [result.height, result.width, result.depth])
    # img = tf.reshape(img, [result.depth, result.height, result.width])
    # # Convert from [depth, height, width] to [height, width, depth].
    # img = tf.transpose(img, [1, 2, 0])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    result.uint8image = img
    return result.uint8image, result.label

def create_record_shuffle():
    writers = [tf.python_io.TFRecordWriter(file_name % i) for i in range(6)]
    index_map = {}
    for index in range(len(classes)):
        index_map.setdefault(index, 0)
    class_num = len(classes)

    total_num = 20000
    i = 0
    writer_index = 0
    while True:
        index = random.randrange(class_num)
        img_index = index_map[index]

        name = classes[index]
        class_path = orig_picture + "/" + name + "/"
        img_name = str(img_index) + ".jpg"
        img_path = class_path + img_name

        index_map[index] += 1
        if (not os.path.exists(img_path)):
            continue
        else:
            i += 1
            img2 = Image.open(img_path)
            img = img2.resize((heigth, width))  # 设置需要转换的图片大小
            img_raw = img.tobytes()  # 将图片转化为原生bytes
            l = len(img_raw)
            if l != heigth * width * channel:
                continue
            print(index, l, i)
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writers[writer_index].write(example.SerializeToString())
            if i >= total_num:
                i = 0
                writer_index += 1

        if writer_index >= len(writers):
            break

    for i in range(6):
        writers[i].close()


# =======================================================================================
if __name__ == '__main__':
    create_record_shuffle()
#     batch = read_and_decode(file_name)
#     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#
#     with tf.Session() as sess:  # 开始一个会话
#         sess.run(init_op)
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#
#         for i in range(num_samples):
#             read_input = batch  # 在会话中取出image和label
#             example, lab = sess.run(read_input)
#             img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
#             img.save(gen_picture + '/' + str(i) + 'samples' + str(lab) + '.jpg')  # 存下图片;注意cwd后边加上‘/’
#             print(example, lab)
#         coord.request_stop()
#         coord.join(threads)
#         sess.close()
#
# # ========================================================================================

