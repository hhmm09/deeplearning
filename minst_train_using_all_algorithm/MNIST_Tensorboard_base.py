import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean' , mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev' , stddev)
        tf.summary.scalar('max' , tf.reduce_max(var))
        tf.summary.scalar('min' , tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


#设置占位符
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x_input')
    y = tf.placeholder(tf.float32,[None,10],name='y_inpute')
    keep_prob = tf.placeholder(tf.float32,)
'''
输入数据格式[none,784] 
输入层 xw ([none,784]*[784,10]) =>[none,10]  b[1,10]
'''
#设置模型参数
#******************************************************************
with tf.name_scope('layer'):
    with tf.name_scope('layer1'):
#输入层以及设置dropout
        with tf.name_scope('wights1'):         
            w1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1),name='w1')
            # variable_summaries(w1)
        with tf.name_scope('biases1'):
            b1 = tf.Variable(tf.zeros([1,2000])+0.1,name='b1')
            # variable_summaries(b1)
        L1 = tf.tanh(tf.matmul(x,w1)+b1)
        with tf.name_scope('L1OUTPUT'):
            L1_Dropout = tf.nn.dropout(L1,keep_prob)
    with tf.name_scope('layer2'):
        with tf.name_scope('wights2'):
            w2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1),name='w2')
            # variable_summaries(w2)
        with tf.name_scope('biases2'):
            b2 = tf.Variable(tf.zeros([1,2000])+0.1,name='b2')
            # variable_summaries(b2)
        L2 = tf.tanh(tf.matmul(L1_Dropout,w2)+b2)
        with tf.name_scope('L2OUTPUT'):
            L2_Dropout = tf.nn.dropout(L2,keep_prob)
    with tf.name_scope('layer3'):
        with tf.name_scope('wights3'):
            w3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1),name='w3')
            # variable_summaries(w3)
        with tf.name_scope('biases3'):
            b3 = tf.Variable(tf.zeros([1,1000])+0.1,name='b3')
            # variable_summaries(b3)
        L3 = tf.tanh(tf.matmul(L2_Dropout,w3)+b3)
        with tf.name_scope('L3OUTPUT'):
            L3_Dropout = tf.nn.dropout(L3,keep_prob)
    with tf.name_scope('layer4'):
        with tf.name_scope('wights4'):
            w4 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1),name='w4')
            variable_summaries(w4)
        with tf.name_scope('biases4'):
            b4 = tf.Variable(tf.zeros([1,10])+0.1,name='b4')
            variable_summaries(b4)
        with tf.name_scope('L4OUTPUT_LAST'):
            pre = tf.nn.softmax(tf.matmul(L3_Dropout,w4)+b4)

#******************************************************************


'''
w1 = tf.Variable(tf.truncated_normal([784,200],stddev=0.1))
b1 = tf.Variable(tf.zeros([1,200])+0.1)
L1 = tf.sigmoid(tf.matmul(x,w1)+b1)
L1_Dropout = tf.nn.dropout(L1,keep_prob)

w2 = tf.Variable(tf.random_normal([200,200]))
b2 = tf.Variable(tf.zeros([1,200]))
L2 = tf.sigmoid(tf.matmul(L1_Dropout,w2)+b2)
L2_Dropout = tf.nn.dropout(L2,keep_prob)

w3 = tf.Variable(tf.random_normal([200,10]))
b3 = tf.Variable(tf.zeros([1,10]))
pre = tf.nn.softmax(tf.matmul(L2_Dropout,w3)+b3)
'''
#**********************************************************
# w1 = tf.Variable(tf.truncated_normal([784,10],stddev=0.1))
# b1 = tf.Variable(tf.zeros([1,10])+0.1)
# L1 = tf.sigmoid(tf.matmul(x,w1)+b1)
# L1_Dropout = tf.nn.dropout(L1,keep_prob)

# w2 = tf.Variable(tf.random_normal([10,10]))
# b2 = tf.Variable(tf.zeros([1,10]))
# pre = tf.nn.softmax(tf.matmul(L1_Dropout,w2)+b2)

#minist数据抓取量
batch_size= 100
batch_time = int(mnist.train.num_examples/batch_size)

#反向传播，损失函数,优化器
#二次代价函数
# loss = tf.reduce_mean(tf.square(y-pre))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pre))
    tf.summary.scalar('loss',loss)
#AdamOptimizer学习率不能太高
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#变量初始化
init = tf.global_variables_initializer()
with tf.name_scope('accuracy'):
#定义正确率
    with tf.name_scope('accuracy_predictioncy'):
        accuracy_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(tf.nn.softmax(pre),1))
    with tf.name_scope('accuracy_rate'):   
        accuracy_rate = tf.reduce_mean(tf.cast(accuracy_prediction,tf.float32))
        tf.summary.scalar('accuracy_rate',accuracy_rate)

merged = tf.summary.merge_all()
#开启绘画
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(20):
        for _ in range(batch_time):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,optimizer], feed_dict={x: batch_xs , y: batch_ys,keep_prob:1.0})
        writer.add_summary(summary,epoch)
        test_acc_rate=sess.run(accuracy_rate,feed_dict={x: mnist.test.images, y: mnist.test.labels , keep_prob:1.0})       
        train_acc_rate=sess.run(accuracy_rate,feed_dict={x: mnist.train.images, y: mnist.train.labels , keep_prob:1.0})
        print('epoch: ' + '%04d  '% (epoch+1)+ 'test_acc_rate: ' +str(test_acc_rate)+'    train_acc_rate'+str(train_acc_rate))
            
 