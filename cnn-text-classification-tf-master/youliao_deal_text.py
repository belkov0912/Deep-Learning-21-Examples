# -*- coding:utf-8 -*-
'''
@author:Levy
@file:dealtext.py
@time:2018/9/1812:22
'''

import jieba
import os
import json
import codecs
import data_helpers
import traceback


basedir ='data/youliao_raw_data/'
stopfilepath = 'stop_words_ch_utf8.txt'
labels = []
file_pointers = {}

stopwords=set()
stopwords.add(' ')

for l in open(stopfilepath, 'r',encoding='UTF-8').readlines():
    stopwords.add(l)

def deal_json(line):
    try:
        class YouliaoRecord(object):
            pass

        line = line.split('zjb-')[1]
        # text = line#.encode('utf-8').decode('utf-8')#.encode('utf-8')
        decode_line = json.loads(line)
        if 'videos' in decode_line:
            return None
        content = decode_line['content']
        title = decode_line['title']
        x = title + '' + content
        # print(content)
        # print('-' * 20)
        x = data_helpers.clean_html(x)
        # print(content2)
        # print('-' * 20)
        seg_text = jieba.cut(x.replace('\t',' ').replace('\n',' '))
        seg_text = filter(lambda x:x not in stopwords ,seg_text)

        result = YouliaoRecord()
        result.x = ' '.join(seg_text)
        category = decode_line['category']
        category = category.split('-')[0]
        if category not in labels:
            labels.append(category)
        result.y = labels.index(category)
        return result
    except Exception as e:
        traceback.print_exc()
        return None

def write_file(x, y):
    if y not in file_pointers:
        writer = open('./data/youliao_train/'+str(labels[y])+'.txt','w',encoding='UTF-8')
        file_pointers[y] = writer

    file_pointers[y].write(x + '\n')


def flush_files():
    for f in file_pointers.values():
        f.flush()

def create_data():
    catenum = 0
    catenum += 1
    # files = os.listdir(basedir)
    files = ['train_data.txt']
    count = 0
    for fileName in files:
        try:
            filepath = basedir + fileName
            with open(filepath,'r',encoding='UTF-8') as fr:
                for line in fr:
                    result = deal_json(line)
                    if result != None:
                        write_file(result.x, result.y)
            count += 1
        except Exception as e:
            traceback.print_exc()
            continue

    flush_files()


if __name__ == '__main__':
    create_data()
