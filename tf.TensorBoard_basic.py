# -*- coding: utf-8 -*-

import tensorflow as tf

a = tf.constant(3.0)
b = tf.constant(5.0)
c = a * b

# tensorboard에 point라는 이름으로 표시됨
c_summary = tf.summary.scalar('point', c)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    # 초기화를 우선적으로 해줘야함.
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./board/sample_1', sess.graph)

    result = sess.run([merged])
    # sess.run(tf.global_variables_initializer())
    # 예제 사이트에선 여기에 init이 있었는데 tensorboard는 작동하나 도표는 안나옴.

    writer.add_summary(result[0])