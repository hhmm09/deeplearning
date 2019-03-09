import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data



LOG_DIR = 'minimalsample'
NAME_TO_VISUALISE_VARIABLE = "mnistembedding"
TO_EMBED_COUNT = 500

path_for_mnist_sprites = os.path.join(LOG_DIR,'mnistdigits.png')
path_for_mnist_metadata = os.path.join(LOG_DIR,'metadata.tsv')
#载入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
batch_xs, batch_ys = mnist.train.next_batch(TO_EMBED_COUNT)

embedding_var = tf.Variable(batch_xs, name=NAME_TO_VISUALISE_VARIABLE)
summary_writer = tf.summary.FileWriter(LOG_DIR)

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


#
def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    
    return spriteimage
#从[1,784]向量转化成[28,28]矩阵
def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits,(-1,28,28))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits
#输入占位符，dropout，学习率
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None], name='y_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    lr = tf.Variable(0.001, dtype=tf.float32, name='learning_rate')

#layer层
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
            z=tf.matmul(L1_Dropout,W2)+b2
        with tf.name_scope('L2OUTPUT_LAST'):

            prediction = tf.nn.softmax(z)



 #损失函数和优化器           
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z))
    tf.summary.scalar('loss', loss)
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
#准确率
with tf.name_scope('train'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)





#这里我们今天直接开启session，替换之前在后面用的with tf.Session() as sess
sess  =  tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
projector_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
#embedding配置
config  =  projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Specify where you find the metadata
embedding.metadata_path = 'metadata.tsv' #'metadata.tsv'

# Specify where you find the sprite (we will create this later)
embedding.sprite.image_path = 'mnistdigits.png' #'mnistdigits.png'
embedding.sprite.single_image_dim.extend([28,28])

# Say that you want to visualise the embeddings
projector.visualize_embeddings(summary_writer, config)


#汇总
merged = tf.summary.merge_all()

#保存模型
saver = tf.train.Saver()

to_visualise = batch_xs
to_visualise = vector_to_matrix_mnist(to_visualise)
to_visualise = invert_grayscale(to_visualise)
#保存图片
sprite_image = create_sprite_image(to_visualise)
plt.imsave(path_for_mnist_sprites,sprite_image,cmap='gray')
plt.imshow(sprite_image,cmap='gray')
#写入metadata
with open(path_for_mnist_metadata,'w') as f:
    f.write("Index\tLabel\n")
    for index,label in enumerate(batch_ys):
        f.write("%d\t%d\n" % (index,label))


for epoch in range(300):
    sess.run(tf.assign(lr, 0.001 * (0.95 ** (epoch / 50))))
    batch_xs, batch_ys = mnist.train.next_batch(100)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0}, options=run_options, run_metadata=run_metadata)
    summary_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
    summary_writer.add_summary(summary, epoch)
    

# for epoch in range(300):
#     sess.run(tf.assign(lr, 0.001 * (0.95 ** (epoch / 50))))
#     batch_xs, batch_ys = mnist.train.next_batch(500)
#     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#     run_metadata = tf.RunMetadata()
#     summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0}, options=run_options, run_metadata=run_metadata)
    

#     if epoch % 100 == 0:
#         test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
#         learning_rate = sess.run(lr)
#         print("Iter" + str(epoch) + ", Testing accuracy:" + str(test_acc) + ", Learning rate:" + str(learning_rate))
saver.save(sess,os.path.join(LOG_DIR, "model.ckpt"), 1)
sess.close()#因为这次没有用with tf.Session() as less，需要在最后加上本条语句