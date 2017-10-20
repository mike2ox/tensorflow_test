# 머신러닝 학습의 Hello World 와 같은 MNIST(손글씨 숫자 인식) 문제를 신경망으로 풀어봅니다.
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# one_hot = True ?
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

##
# 옵션 설정
##
learning_rate = 0.001
total_epoch = 30
batch_size = 128
detail_dir_route = "/GRU/LR/3e-1"

# RNN 은 순서가 있는 자료를 다루므로,
# 한 번에 입력받는 갯수와, 총 몇 단계로 이루어져있는 데이터를 받을지를 설정해야합니다.
# 이를 위해 가로 픽셀수를 n_input 으로, 세로 픽셀수를 입력 단계인 n_step 으로 설정하였습니다.
n_input = 28
n_step = 28
n_hidden = 128
n_class = 10

###
# 신경망 모델 구성
###

global_step = tf.Variable(0, trainable=False, name='global_step')

with tf.variable_scope('input_data'):
    X = tf.placeholder(tf.float32, [None, n_step, n_input])
    Y = tf.placeholder(tf.float32, [None, n_class])

    W = tf.Variable(tf.random_normal([n_hidden, n_class]))
    b = tf.Variable(tf.random_normal([n_class]))

    tf.summary.histogram("X", X)
    tf.summary.histogram("Weight", W)
    tf.summary.histogram("bias", b)

with tf.variable_scope('make_RNNcell'):
    # RNN 신경망을 구성할 Cell을 생성
    # BasicRNNCell,BasicLSTMCell,GRUCell

    cell = tf.nn.rnn_cell.GRUCell(n_hidden)

with tf.variable_scope('make_RNN'):
    # RNN 신경망을 생성합니다
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    tf.summary.histogram("Outputs", outputs)

with tf.variable_scope('transpose'):
    # 결과를 Y의 형식으로 바꿔주기 위해 outputs의 형태를 변경한다
    outputs = tf.transpose(outputs, [1, 0, 2])

    # 왜 -1??
    outputs = outputs[-1]
    model = tf.matmul(outputs, W) + b

    tf.summary.histogram("Model", model)

with tf.variable_scope('opt'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.variable_scope('accuracy'):
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# tf.summary.scalar 를 이용해 수집하고 싶은 값들을 지정할 수 있습니다.
tf.summary.scalar('cost', cost)
tf.summary.scalar('accuracy',accuracy)

#
# 신경망 모델 학습
#
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter('./logs' + detail_dir_route, sess.graph)
writer.add_graph(sess.graph)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # X 데이터를 RNN input에 맞게 [batch_size, n_step, n_input] 형태로 변환합니다.
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

        summary = sess.run(merged, feed_dict={X: batch_xs, Y: batch_ys})
        writer.add_summary(summary, global_step=sess.run(global_step))

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

    print('최적화 완료!')

    #########
    # 결과 확인
    ######
    test_batch_size = len(mnist.test.images)
    test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
    test_ys = mnist.test.labels

    print('정확도:', sess.run(accuracy,
                           feed_dict={X: test_xs, Y: test_ys}))



