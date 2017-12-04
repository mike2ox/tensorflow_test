import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)


total_epoch = 200
batch_size = 100
learning_rate = 0.0002

n_hidden = 256
n_input = 28 * 28
n_noise = 128


X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

# 가짜 생성 모델
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# input이 진짜인지 가짜인지 구별하는 모델
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))


def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)

    return output

def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)

    return output

def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

# 노이즈 data인 z를 이용해 랜덤 이미지 생성(G)
G = generator(Z)

# G를 기반으로 가짜 이미지 정보를 가지고 있는 D_gene 생성
D_gene = discriminator(G)

# 실제 데이터인 X를 기반으로 D_real을 생성
D_real = discriminator(X)

# minmaxV(D,G)를 구현(D는 구별을 최대로, G는 구별이 안되는걸 최대로)
# D의 입장에서 minmax value function 결과 최대화 하기(reduce_mean을 통해 음수가 나온 값을 양수로 바꿔주면 됨)
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
# G의 입장에서 minmax value function 결과 최소화 하기(D(x)대신에 D(G(z))가 max가 되도록 구현, 얘도 양수로 바꿔줌)
loss_G = tf.reduce_mean(tf.log(D_gene))
# 같은 의미 : loss_G = tf.reduce_mean(tf.log())

D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

    print('epoch:', '%04d' % epoch,
          'D loss: {:4}'.format(loss_val_D),
          'G loss: {:4}'.format(loss_val_G))

    if epoch == 0 or (epoch + 1) % 3 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('041217/samples2/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료!')



