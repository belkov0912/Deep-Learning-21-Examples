import numpy as np
import re
import os
import tensorflow as tf

def load_data_and_labels(dir):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    files = os.listdir(dir)
    catenum=len(files)
    cateindex=0
    tmpres=[]
    tmpdate=[]
    for fileName in files:
        # examples = list(open(dir+fileName, "r",encoding='UTF-8').readlines())
        examples = tf.data.TextLineDataset(dir + fileName)
        # examples = [s.strip() for s in examples]
        examples = examples.map(parse_string)

        #tx_text = [clean_str(sent) for sent in tx_text]
        tmpdate.append(examples)
        tmplabel = [0]*catenum
        tmplabel[cateindex]=1
        # Generate labels
        labels = [tmplabel for _ in tf.range(examples.output_shapes[0])]
        cateindex =cateindex+1
        tmpres.append(labels)
    y = np.concatenate(tmpres, 0)
    x_text=np.concatenate(tmpdate,0)
    return [x_text, y]

def parse_string(x):
    x = tf.strings.strip(x)
    x = tf.strings.split([x], ' ').values[0:200]
    x = tf.strings.join([x], ' ')
    return x

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
