import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#图片数量
image_num = 500
#文件路径
DIR = "C://Users/Mars/Desktop/wokstation/"#设置文件路径（这是我的程序文件位置，大家自行做修改），之后生成metadata文件时会用到
#载入图片
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')
#tf.stack(mnist.test.images[:image_num])中的image_num可以控制测试图片个数，最大为10000，我这里取值为10000
# 批次
n_batch = 100

sess = tf.Session()
#这里我们今天直接开启session，替换之前在后面用的with tf.Session() as sess

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)  ##直方图
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    lr = tf.Variable(0.001, dtype=tf.float32, name='learning_rate')
# with tf.name_scope('layer'):
#     with tf.name_scope('Input_layer'):
#         with tf.name_scope('W1'):
#             W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1), name='W1')
#             variable_summaries(W1)
#         with tf.name_scope('b1'):
#             b1 = tf.Variable(tf.zeros([500]) + 0.1, name='b1')
#             variable_summaries(b1)
#         with tf.name_scope('L1'):
#             L1 = tf.nn.relu(tf.matmul(x, W1) + b1, name='L1')
#             L1_drop = tf.nn.dropout(L1, keep_prob)
#             prediction = tf.nn.softmax(tf.matmul(L1_drop, W2) + b2)



with tf.name_scope('layer'):
    with tf.name_scope('layer1'):
        with tf.name_scope('W1'):
            W1 = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1), name='W1')
            variable_summaries(W1)
        with tf.name_scope('b1'):
            b1 = tf.Variable(tf.zeros([1,10])+0.1)
            variable_summaries(b1)
        with tf.name_scope('L1'):
            L1 = tf.sigmoid(tf.matmul(x,W1)+b1)
            L1_Dropout = tf.nn.dropout(L1,keep_prob)
    with tf.name_scope('layer2'):
        with tf.name_scope('W2'):
            W2 = tf.Variable(tf.random_normal([10,10]))
        with tf.name_scope('b2'):
            b2 = tf.Variable(tf.zeros([1,10]))
        with tf.name_scope('L2OUTPUT_LAST'):
            prediction = tf.nn.softmax(tf.matmul(L1_Dropout,W2)+b2)




with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.name_scope('train'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
sess.run(init)
merged = tf.summary.merge_all()
# 生成metadata文件
with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')

#下段代码，按照Tensorflow文档进行操作
projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path ='C:/Users/Mars/Desktop/wokstation/projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28, 28])
#将下载下来的数据集图片以28x28的像素进行分隔，分隔成一个个数字图片
projector.visualize_embeddings(projector_writer, config)

for epoch in range(300):
    sess.run(tf.assign(lr, 0.001 * (0.95 ** (epoch / 50))))
    batch_xs, batch_ys = mnist.train.next_batch(n_batch)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0}, options=run_options, run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
    projector_writer.add_summary(summary, epoch)
    

    if epoch % 100 == 0:
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        learning_rate = sess.run(lr)
        print("Iter" + str(epoch) + ", Testing accuracy:" + str(test_acc) + ", Learning rate:" + str(learning_rate))

saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=1001)
projector_writer.close()
sess.close()#因为这次没有用with tf.Session() as less，需要在最后加上本条语句
