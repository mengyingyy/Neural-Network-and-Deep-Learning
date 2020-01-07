# Demo - train the style transfer network & use it to generate an image
# This code is modified from https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/main.py
# We modified his input and output/generation files to finish our task

from __future__ import print_function

from train import train
from generate import generate
from utils import list_images

import matplotlib.pyplot as plt


IS_TRAINING = False

VGG_PATH  = './imagenet-vgg-19-weights.npz'

# format: {'style': [content_weight, style_weight, tv_weight]}
STYLES = {
    # 'illus':             [1.0,  15.0, 1e-2],
    # 'echo':              [1.0,  8.0, 1e-2],
    'jojo':              [1.0,  3.0, 1e-2]
    # 'hinata':          [1.0,  20.0, 1e-2]
    # 'flower':          [1.0,  10.0, 1e-2],
    # 'scream':          [1.0,  60.0, 1e-2],
    # 'denoised_starry': [1.0,  16.0, 1e-2],
    # 'starry_bright':   [1.0,   6.0, 1e-2],
    # 'rain_princess':   [1.0,   8.0, 1e-2],
    # 'woman_matisse':   [1.0,  20.0, 1e-2],
    # 'mosaic':          [1.0,   5.0,  0.0],
}


def main():

    if IS_TRAINING:

        content_targets = list_images('./contents') # path to training dataset

        for style in list(STYLES.keys()):

            print('\nBegin to train the network with the style "%s"...\n' % style)

            content_weight, style_weight, tv_weight = STYLES[style]

            style_target = 'images//style//' + style + '.jpg'
            model_save_path = 'models//' + style + '.ckpt-done'

            content_loss, style_loss, tv_loss, total_loss = train(content_targets, style_target, content_weight, style_weight, tv_weight,
                vgg_path=VGG_PATH, save_path=model_save_path)
            x_axis = [i*100 for i in range(len(content_loss))]
            plt.plot(x_axis, content_loss, label='content_loss')
            plt.plot(x_axis, style_loss, label='style_loss')
            plt.plot(x_axis, total_loss, label='total_loss')

            plt.legend()
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.savefig('losses_'+style+'.png')
            plt.clf()

            print('\nSuccessfully! Done training style "%s"...\n' % style)

        print('Successfully finish all the training...\n')
    else:

        for style in list(STYLES.keys()):

            print('\nBegin to generate pictures with the style "%s"...\n' % style)

            model_path = 'models/' + style + '.ckpt-done'
            output_save_path = 'outputs'

            content_targets = list_images('images/content')
            generated_images = generate(content_targets, model_path, save_path=output_save_path, 
                prefix=style + '-')

            print('\ntype(generated_images):', type(generated_images))
            print('\nlen(generated_images):', len(generated_images), '\n')


if __name__ == '__main__':
    main()

