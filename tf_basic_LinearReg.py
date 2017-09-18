import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
print(X)
print(Y)

hypo = W*X + b

cost = tf.reduce_mean(tf.square(hypo - Y))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)

train_op = opt.minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X:x_data,Y:y_data})

        print(step, cost_val, sess.run(W),sess.run(b))


    print('\n===test===')
    print("X : 5, Y :",sess.run(hypo, feed_dict={X:5}))
    print("X : 2.5, Y :", sess.run(hypo, feed_dict={X: 2.5}))
