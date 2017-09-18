import tensorflow as tf

# tf.placeholder : 계산을 실행할때 입력받는 변수로 사용함.
X = tf.placeholder(tf.float32, [None,3])
print(X)

x_data = [[1,2,3],[4,5,6]]

#Variable : 그래프 opt할때 필요한 변수들, random_normal : 정규분포 랜덤값으로 초기화
W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

#mat* 형식의 함수는 행렬 계산을 수행해 주는 것이다.
expr = tf.matmul(X,W) + b
sess = tf.Session()

#처음에 세션을 만들면 tf의 전역변수들을 초기화 해줘야 한다.
sess.run(tf.global_variables_initializer())

print('===x_data===')
print(x_data)
print('===W===')
print(sess.run(W))
print('===b===')
print(sess.run(b))
print('===expr===')

print(sess.run(expr, feed_dict={X: x_data}))

sess.close()
