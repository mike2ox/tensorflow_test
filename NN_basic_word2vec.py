import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


font_name = matplotlib.font_manager.FontProperties(
    fname="C:/Windows/Fonts/Gulim.ttc"
).get_name()
matplotlib.rc('font',family = font_name)

sentences = ["나 고양이 좋다",
             "나 강아지 좋다",
             "나 동물 좋다",
             "강아지 고양이 동물",
             "여자친구 고양이 강아지 좋다",
             "고양이 생선 우유 좋다",
             "강아지 생선 싫다 우유 좋다",
             "강아지 고양이 눈 좋다",
             "나 여자친구 좋다",
             "여자친구 나 싫다",
             "여자친구 나 영화 책 음악 좋다",
             "나 게임 만화 애니 좋다",
             "고양이 강아지 싫다",
             "강아지 고양이 좋다"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))

word_dict = {w : i for i, w in enumerate(word_list)}
word_index = [word_dict[word] for word in word_list]


skip_grams = []

for i in range(1, len(word_index) - 1) :
    target = word_index[i]
    context = [word_index[i-1], word_index[i+1]]

    for w in context:
        skip_grams.append([target,w])

def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])
        random_labels.append([data[i][1]])

    return random_inputs, random_labels


training_epoch = 300
learning_rate = 0.1

batch_size = 20
embedding_size = 2

num_sampled = 15

voc_size = len(word_list)


inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size,1])

embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))

selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

nce_weight = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(nce_weight, nce_biases, labels, selected_embed, num_sampled, voc_size))
train_op =tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(1, training_epoch+1):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

        _, loss_val = sess.run([train_op,loss],
                               feed_dict={inputs : batch_inputs,
                                          labels : batch_labels})

        if step % 10 == 0:
            print("loss at step", step ,":", loss_val)

    trained_embeddings = embeddings.eval()

for i, labels in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x,y)
    plt.annotate(labels, xy=(x,y),xytext=(5,2),
                 textcoords='offset points', ha='right',va='bottom')

plt.show()




