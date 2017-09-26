#텐서플로우를 import해서 쓸수있도록 설정
import tensorflow as tf

#x,y데이터를 텐서에 기입. 정답을 아는 상황.
x_data = [1,2,3]
y_data = [1,2,3]

# W,b를 랜덤하게 받는다. 1개만 받고, -1 ~ 1 사이에서 뽑히도록 한다
# viriable : 텐서플로우가 사용하는 변수이다. trainable variable이라고 생각. 우리가 설정하지 않음.
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))


#placeholder는 뒤에 feed_dict을 통해서 값을 넣을수 있도록 해주는 메소드. (형식,이름)
X = tf.placeholder(tf.float32,name='X')
Y = tf.placeholder(tf.float32, name='Y')
#print(X)
#print(Y)

hypo = W*X + b

cost = tf.reduce_mean(tf.square(hypo - Y))

#최적화 함수방식(기울기 감소 함수)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)

#cost를 최소화 한다
train_op = opt.minimize(cost)

#with를 사용해서 tf.close()를 따로 작성안해줘도 됨.
with tf.Session() as sess:

    #variable이 있을 경우 초기화 작업을 run할때 해줘야 한다.
    sess.run(tf.global_variables_initializer())
    print(sess.run(W), sess.run(b))

    print("step | cost_val | W | b")

    for step in range(100):
        #_는 별로 의미 없음을 표현하는 것으로 cost를 최소화 할때의 값이므로 최저값만 찾으면 되기에 굳이 출력할 필요가 없어서 이름을 붙여놓지 않음.
        _, cost_val = sess.run([train_op, cost], feed_dict={X:x_data,Y:y_data})

        #for 단계
        print(step, cost_val, sess.run(W),sess.run(b))



    print('\n===test===')
    print("X : 5, Y :",sess.run(hypo, feed_dict={X:5}))
    print("X : 2.5, Y :", sess.run(hypo, feed_dict={X: 2.5}))


'''
1. Variable안에 shape = [none]입력 : print test다음에 sess.run에서 variable 에러가 발생
2. learning rate를 변경 
    1) 0.5일때 cost_val가 커지고 W,b의 절대값 또한 커짐. 얼마 못가서 측정 불가라고 뜸
    2) 0.01일때 cost_val가 작아지고는 있으나 엄청 작은 값으로는 수렴하지 않음. 실제로 X=5일때는 Y값이 4보다도 작게 출력됨
3. range 변경(100 -> 1000)
    1) range만 변경 : 645쯤 되서는 아에 cost_val을 0.0으로 확정 지음.
    2) range, learning rate변경(0.2) : 이것도 2-1과 같은 결과
    3) range, learning rate변경(0.09) : 2-2의 과정을 지나서 점점 cost_val이 작아지기는 하나 3-1처럼 확정 짓지는 않음
4. global_variables_init 생략 : 초기화 안된 value를 사용한다는 error발생(FailedPreconditionError).
    1) init하기 전 W, s 출력 : Tensor("Variable/read:0", shape=(1,), dtype=float32) Tensor("Variable_1/read:0", shape=(1,), dtype=float32)
    2) init하기 전 sess.run(W), sess.run(s) 출력 : 초기화 안된 value를 사용한다는 error발생(FailedPreconditionError).
    3) init하고나서 W, s, sess.run(W), sess.run(s) 출력 : W, s는 똑같음. 그냥 형식만 출력됨
                                                        sess.run으로는 W,s에 랜덤한 수가 입력됨.
5. sess.run안에 []를 없앴을 경우 : typeerror발생. 
                ()로 바꿨을 경우 : 정상 작동함.  {}로 바꿧을 경우  : 오류남

'''

