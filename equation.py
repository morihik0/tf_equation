import tensorflow as tf
import numpy as np

# random number 0.0~1.0
x = np.random.rand(100).astype(np.float32)
# y = 2x^2 + 18
y = 2 * x * x +18

# model
with tf.name_scope('model'):
    w = tf.Variable(tf.zeros([1]))
    b = tf.Variable(tf.zeros([1]))
    y_model = w * x * x + b

# loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square( y_model - y))

# train
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer().minimize(loss)

# distance
with tf.name_scope('distance'):
    dist = tf.reduce_mean(y_model - y)

tf.summary.scalar('distance', dist)
merged = tf.summary.merge_all()

# session
with tf.Session() as sess:
    writer = tf.summary.FileWriter('log', sess.graph)
    sess.run(tf.global_variables_initializer())

    #learn
    for step in range(40000):
        sess.run(train)

        summary, distlog = sess.run([merged,dist])
        writer.add_summary(summary, step)

        if step % 1000 == 0:
            print(step,sess.run(w),sess.run(b))


