import numpy as np
import re
import os
import tensorflow as tf

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(dir, max_doc_length):
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
        examples = list(open(dir+fileName, "r",encoding='UTF-8').readlines())
        examples = [s.strip() for s in examples]
        examples = [" ".join(x.split(" ")[0:max_doc_length]) for x in examples]

    # Split by words
        tx_text =  examples
        #tx_text = [clean_str(sent) for sent in tx_text]
        tmpdate.append(tx_text)
        tmplabel = [0]*catenum
        tmplabel[cateindex]=1
    # Generate labels
        labels = [tmplabel for _ in examples]
        cateindex =cateindex+1
        tmpres.append(labels)
    y = np.concatenate(tmpres, 0)
    x_text=np.concatenate(tmpdate,0)
    return [x_text, y]


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
