import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)




batch_size = 10
n_batch =mnist.train.num_examples//batch_size

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(initial)
def biass_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    # Given an input tensor of shape [batch, in_height, in_width, in_channels]
    # and a filter / kernel tensor of shape[filter_height, filter_width, in_channels, out_channels]
    # horizontal and vertices strides, strides = [1, stride, stride, 1]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
def avg_pool_7x7(x):
  return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1],
                        strides=[1, 7, 7, 1], padding='SAME')

#set module placeholder
with tf.name_scope('input'):    
    x = tf.placeholder(dtype=tf.float32,shape=[None,784],name='input_image')
    y = tf.placeholder(dtype=tf.float32,shape=[None,10],name='lables')

x_images = tf.reshape(x,shape=[-1,28,28,1])
#初始化第一个卷积层的权值和偏置
W_conv1 = weight_variable(shape=[5,5,1,32])#5*5的卷积窗口，1通道数出，32通道输出
b_conv1 = biass_variable(shape=[32])#偏置值，共有32个feature map
#执行卷积操作再加上偏置，同过激励函数relu
h_conv1 = tf.nn.relu(conv2d(x_images,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#初始化第二个卷积层的权值和偏置
W_conv2 = weight_variable(shape=[5,5,32,64])#从32通道拓展层64通道
b_conv2 = biass_variable(shape=[64])

#执行卷积
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = biass_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

nt_hpool3=avg_pool_7x7(h_conv3)#64
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
y_conv=tf.nn.softmax(nt_hpool3_flat)

#由于conv2d的padding设置为SAME 所以是同卷积操作
#而池化层strides为2x2所以没经过一次池化维度除以2

'''
#初始化全连接层
W_fc1 = weight_variable([7*7*64,100])#上一层有7*7*64个神经元，这一层有1024个神经元
b_fc1 = biass_variable([100])


#把池化层2的输出化为一维
h_pool2_flat = tf.reshape(h_pool2,shape=[-1,7*7*64])
#求第一个全部连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)



W_fc2 = weight_variable([100,10])
b_fc2 = biass_variable([10])

prediction = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
'''
# pre_sorfmax = tf.nn.softmax(prediction)
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))

optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# correct_prediction = tf.equal(tf.argmax(pre_sorfmax,1),tf.argmax(y,1))

# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(200):
        for bacth in range(n_batch):
            batch_xs,batch_ys =mnist.train.next_batch(batch_size)
            sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Iter :' + epoch + 'accuracy :' + acc )


'''
# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200000):#20000
      batch_xs,batch_ys =mnist.train.next_batch(100)#50
      if i%20 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch_xs, y: batch_ys})
        print( "step %d, training accuracy %g"%(i, train_accuracy))
        optimizer.run(feed_dict={x: batch_xs, y: batch_ys})
    

    