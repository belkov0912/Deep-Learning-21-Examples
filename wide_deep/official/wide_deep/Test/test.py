
_CSV_COLUMNS = [
    'gender', 'age', 'createtime', 'addtime', 'clickmax',
    'clickcount', 'showcount', 'likemax', 'likecount', 'dislikecount',
    'showgoodpercent', 'allclicknum', 'allshownum', 'sharenum', 'favoritenum',
    'cateid', 'topicid', 'keywordid', 'cf_score', 'topic_score',
    'keyword_score','topicctr_score', 'topic_score_short', 'topicctr_score_short', 'dt_score',
    'toutiaotaste_score', 'apptaste_score','label'
]
with open("/Users/jiananliu/work/machinelearn/dl/wide_deep/data/train.data", "r") as f:
    i = 0
    for line in f:
        sp = line.split(",", -1)
        print(dict(zip(_CSV_COLUMNS, sp)))
        # print(line)
        i += 1
        if i >= 5:
            break

import tensorflow as tf
with tf.Session() as sess:
    aa = tf.constant("12")
    bb = tf.string_to_number(aa, tf.int32)
    print(sess.run(bb))
