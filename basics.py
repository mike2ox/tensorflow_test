#텐서플로우를 가져옴
import tensorflow as tf

#tf에 있는 constant를 만들고 그걸 hello라는 하나의 텐서가 생성되는것
hello = tf.constant("Hello, TensorFlow!")

#session을 실행해야 텐서들을 작동시킬수 있다.
sess = tf.Session()
print(sess.run(hello))

#b가 출력되는건 bite stream이라는걸 표현하는 것이므로 문제없음.