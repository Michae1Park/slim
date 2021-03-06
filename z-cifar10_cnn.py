from datasets import cifar10
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.contrib import slim
from preprocessing import inception_preprocessing

#----------------------------
# Displaying Data
#----------------------------
data_dir = '/home/michael/Desktop/learning-tf/cifar10'

# with tf.Graph().as_default(): 
#     dataset = cifar10.get_split('train', data_dir)
#     data_provider = slim.dataset_data_provider.DatasetDataProvider(
#         dataset, common_queue_capacity=32, common_queue_min=1)
#     image, label = data_provider.get(['image', 'label'])
    
#     with tf.Session() as sess:    
#         with slim.queues.QueueRunners(sess):
#             for i in range(4):
#                 np_image, np_label = sess.run([image, label])
#                 height, width, _ = np_image.shape
#                 class_name = name = dataset.labels_to_names[np_label]
                
#                 plt.figure()
#                 plt.imshow(np_image)
#                 plt.title('%s, %d x %d' % (name, height, width))
#                 plt.axis('off')
#                 plt.show()


#----------------------------
# CNN
#----------------------------
def my_cnn(images, num_classes, is_training):  # is_training is not used...
    with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
        net = slim.conv2d(images, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 192)
        net = slim.fully_connected(net, num_classes, activation_fn=None)       
        return net


def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False):
    """Loads a single batch of data.
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])
    
    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
    
    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
          [image, image_raw, label],
          batch_size=batch_size,
          num_threads=1,
          capacity=2 * batch_size)
    
    return images, images_raw, labels


#----------------------------
# Train
#----------------------------
# This might take a few minutes.
train_dir = '/home/michael/Desktop/learning-tf/models/research/slim/weights/cifar10-scratch'
print('Will save model to %s' % train_dir)

# with tf.Graph().as_default():
#     tf.logging.set_verbosity(tf.logging.INFO)

#     dataset = cifar10.get_split('train', data_dir)
#     images, _, labels = load_batch(dataset)
  
#     # Create the model:
#     logits = my_cnn(images, num_classes=dataset.num_classes, is_training=True)
 
#     # Specify the loss function:
#     one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
#     slim.losses.softmax_cross_entropy(logits, one_hot_labels)
#     total_loss = slim.losses.get_total_loss()

#     # Create some summaries to visualize the training process:
#     tf.summary.scalar('losses/Total Loss', total_loss)
  
#     # Specify the optimizer and create the train op:
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#     train_op = slim.learning.create_train_op(total_loss, optimizer)

#     # Run the training:
#     final_loss = slim.learning.train(
#       train_op,
#       logdir=train_dir,
#       number_of_steps=1, # For speed, we just do 1 epoch
#       save_summaries_secs=1)
  
#     print('Finished training. Final batch loss %d' % final_loss)


#----------------------------
# Test
#----------------------------
with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    
    dataset = cifar10.get_split('test', data_dir)
    images, _, labels = load_batch(dataset)
    
    logits = my_cnn(images, num_classes=dataset.num_classes, is_training=False)
    predictions = tf.argmax(logits, 1)

    print(labels)
    print(predictions)