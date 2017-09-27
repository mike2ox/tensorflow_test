import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/",one_hot=True)

# 28*28*1의 image와 Y ?
# 갯수는 none임.
X = tf.placeholder(tf.float32, [None,28,28,1])
Y = tf.placeholder(tf.float32, [None,10])

keep_prob = tf.placeholder(tf.float32)

# [3,3]은 커널(무게 필터)의 크기, 1: 입력값 X의 특성 수 / color, 32 : 필터의 갯수 = 통과했을때 결과물의 갯수
# stddev?
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev= 0.01))

# convolution layer를 통과 시킴 stride : 가운데 2요소가 움직이는 칸의 수 ㅇㅇ
# X : 이미지, W1 : 무게, SAME : 원래 이미지랑 같게 나오도록 해주는 설정.
L1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
L1 = tf.nn.relu(L1)

# ksize에서 가운데 2요소가 필터의 사이즈
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 두번째 무게 필터
# X가 32개이고 필터가 64개
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

# L2의 Conv 형태 = (?,14,14,64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# FC 레이어 : input 7*7*64 --> output 256
# FC을 위해 직전의 Pool 사이즈인 (?,7,7,64)를 참고하여 차원을 줄여줍니다.
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))

# reshape에서 요소들의 의미는?
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

# 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 0~9 레이블인 10개의 출력값을 만듭니다.
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
# 최적화 함수를 RMSPropOptimizer 로 바꿔서 결과를 확인해봅시다. --> Adam보다 정확도가 더 높다.
# optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# batch는 filter(kernal의 수)
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

# range의 범위를 올릴수록 더 정확한 accuracy가 도출됨.
for epoch in range(20):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          keep_prob: 0.7})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

#########
# 결과 확인
######
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                       feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                  Y: mnist.test.labels,
                                  keep_prob: 1}))