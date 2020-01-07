import tensorflow as tf
import image_transform_net as itn
import numpy as np

from loss_net import VGG, preprocess
from utils import get_images

from datetime import datetime

CONTENT = 'relu4_2'
STYLE = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')
IMG_SHAPE = (256, 256, 3)
EPOCHS = 5
batch_size = 4
lr = 2e-3

def train(content_path, style_path, content_weight, style_weight, tv_weight, vgg_path, save_path):
    height, width, channels = IMG_SHAPE
    input_shape = (batch_size, height, width, channels)

    start_time = datetime.now()

    vgg = VGG(vgg_path)

    style_target = get_images(style_path, height, width)
    style_shape = style_target.shape

    style_features = {}
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # style net
    with tf.Session(config=config) as sess:
        style_img = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_net = vgg.forward(preprocess(style_img))

        for layer in STYLE:
            features = style_net[layer].eval(feed_dict={style_img: style_target})
            features = np.reshape(features, [-1, features.shape[3]])

            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    # content net
    with tf.Session(config=config) as sess:
        content_img = tf.placeholder(tf.float32, shape=input_shape, name='content_img')

        content_net = vgg.forward(preprocess(content_img))
        content_features = content_net[CONTENT]

        trans_images = itn.transform(content_img)
        output_net = vgg.forward(preprocess(trans_images))

        # reconstruction loss
        content_size = tf.size(content_features)
        content_loss = tf.nn.l2_loss(output_net[CONTENT] - content_features)*2 / tf.to_float(content_size)

        # style reconstruction loss
        style_losses = []
        for layer in STYLE:
            features = output_net[layer]
            shape = tf.shape(features)
            num_images, height, width, num_filters = shape[0], shape[1], shape[2], shape[3]

            features = tf.reshape(features, [num_images, height * width, num_filters])

            grams = tf.matmul(features, features, transpose_a=True) / tf.to_float(height * width * num_filters)
            style_gram = style_features[layer]

            layer_style_loss = tf.nn.l2_loss(grams - style_gram)*2 / tf.to_float(tf.size(grams))
            style_losses.append(layer_style_loss)

        style_loss = tf.reduce_sum(tf.stack(style_losses))

        # total variation loss
        shape = tf.shape(trans_images)
        height, width = shape[1], shape[2]
        y = tf.slice(trans_images, [0, 0, 0, 0], [-1, height - 1, -1, -1]) - tf.slice(trans_images, [0, 1, 0, 0],
                                                                                       [-1, -1, -1, -1])
        x = tf.slice(trans_images, [0, 0, 0, 0], [-1, -1, width - 1, -1]) - tf.slice(trans_images, [0, 0, 1, 0],
                                                                                      [-1, -1, -1, -1])

        tv_loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

        # overall perceptual losses
        loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss

        # Training step
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        step = 0
        n_batches = len(content_path) // batch_size

        elapsed_time = datetime.now() - start_time
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging.info('Elapsed time for preprocessing before actually train the model: %s' % elapsed_time)
        tf.logging.info('Now begin to train the model...')
        start_time = datetime.now()

        c_loss = []
        s_loss = []
        tv = []
        total_loss = []
        for epoch in range(EPOCHS):

            np.random.shuffle(content_path)

            for batch in range(n_batches):
                # retrive a batch of content_targets images
                content_batch_path = content_path[batch * batch_size:(batch * batch_size + batch_size)]
                content_batch = get_images(content_batch_path, input_shape[1], input_shape[2])

                # run the training step
                sess.run(train_op, feed_dict={content_img: content_batch})

                step += 1

                if step % 1000 == 0:
                    saver.save(sess, save_path, global_step=step)

                is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                if is_last_step or step % 100 == 0:
                    elapsed_time = datetime.now() - start_time
                    _content_loss, _style_loss, _tv_loss, _loss = sess.run(
                        [content_loss, style_loss, tv_loss, loss], feed_dict={content_img: content_batch})

                    tf.logging.info('step: %d,  total loss: %f,  elapsed time: %s' % (step, _loss, elapsed_time))
                    tf.logging.info('content loss: %f,  weighted content loss: %f' % (
                        _content_loss, content_weight * _content_loss))
                    tf.logging.info(
                        'style loss  : %f,  weighted style loss  : %f' % (_style_loss, style_weight * _style_loss))
                    tf.logging.info(
                        'tv loss     : %f,  weighted tv loss     : %f' % (_tv_loss, tv_weight * _tv_loss))
                    tf.logging.info('\n')
                    c_loss.append(_content_loss)
                    s_loss.append(_style_loss)
                    tv.append(_tv_loss)
                    total_loss.append(_loss)

        saver.save(sess, save_path)
        return c_loss, s_loss, tv, total_loss
