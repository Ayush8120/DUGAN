import os
import datetime
import click
import numpy as np
import tqdm
import tensorflow as tf
from deblurgan.utils import load_images, write_log
from deblurgan.losses import  perceptual_loss, ns_generator_loss, euc_dist_keras, wasserstein_loss
#from deblurgan.losses import  perceptual_loss, ns_generator_loss, euc_dist_keras
from deblurgan.model import generator_model, discriminator_model, generator_containing_discriminator_multiple_outputs
from PIL import Image
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
BASE_DIR = 'weights/Rotational_weights/'
tf.compat.v1.disable_eager_execution()

def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)


def train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates=5):
    data = load_images('./Biyani/Separate_Dataset/Rotational_blur_Dataset', n_images)
    y_train, x_train = data['B'], data['A']
    print(x_train.shape)
    print(y_train.shape)

    g = generator_model()
    tf.keras.utils.plot_model(
    g, to_file='g_model[Rot].png', show_shapes=True, show_dtype=True,
    show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96 )
    
    d = discriminator_model()
    tf.keras.utils.plot_model(
    d, to_file='d_model[Rot].png', show_shapes=True, show_dtype=True,
    show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96 )
    
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)
    tf.keras.utils.plot_model(
    d_on_g, to_file='d_on_g_model[Rot].png', show_shapes=True, show_dtype=True,
    show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96 )

    d_opt = Adam(lr= 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08) #HERE
    d_on_g_opt = Adam(lr= 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08) #HERE
    # To resume training from a particular epoch, load weights here
    g.load_weights("/content/drive/MyDrive/deblur-gan/weights/Rotational_weights/48/generator_39_0.h5")
    d.load_weights("/content/drive/MyDrive/deblur-gan/weights/Rotational_weights/48/discriminator_39.h5")

    d.trainable = True
    #d.compile(optimizer=d_opt, loss= ns_generator_loss)
    d.compile(optimizer=d_opt, loss= wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    #loss = [perceptual_loss, ns_generator_loss]
    factor = 1/(320*320*3)
    #loss_weights = [1, 0.000002]
    loss_weights = [100, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True

    #output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))
    #output_true_batch, output_false_batch = np.ones((batch_size, 1)), np.zeros((batch_size, 1))
    output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))
    log_path = './log'
    tensorboard_callback = TensorBoard(log_path)

    for epoch in tqdm.tqdm(range(epoch_num)):
        if epoch > 39 : # if continuing from mid epoch then change here
          permutated_indexes = np.random.permutation(x_train.shape[0])

          d_losses = []
          d_on_g_losses = []
          for index in range(int(x_train.shape[0] / batch_size)):
              print("epoch:--------- "+str(epoch)+"------------batch: "+str(index+1))
              batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
              image_blur_batch = x_train[batch_indexes]
              image_full_batch = y_train[batch_indexes]
              generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)
              # print(generated_images.shape)

              for i in range(critic_updates):
                  print(" critic_update: "+str(i+1))
                  d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                  d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                  d_loss = 0.5 * (d_loss_fake + d_loss_real)
                  d_losses.append(d_loss)
                  print("d_loss: "+str(d_loss))

              d.trainable = False

              d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
              d_on_g_losses.append(d_on_g_loss)

              d.trainable = True
              print("Batch "+str(index) +" Training completed")

          # write_log(tensorboard_callback, ['g_loss', 'd_on_g_loss'], [np.mean(d_losses), np.mean(d_on_g_losses)], epoch_num)
          # print(np.mean(d_losses), np.mean(d_on_g_losses))
          print(np.mean(d_losses), np.mean(d_on_g_losses))

          with open('log.txt', 'a+') as f:
            f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))
          
          save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))
          print("Epoch "+str(epoch) +" Training completed")




@click.command()
@click.option('--n_images', default=-1, help='Number of images to load for training')
@click.option('--batch_size', default=16, help='Size of batch')
@click.option('--log_dir', required=True, help='Path to the log_dir for Tensorboard')
@click.option('--epoch_num', default=2, help='Number of epochs for training')
@click.option('--critic_updates', default=5, help='Number of discriminator training')
def train_command(n_images, batch_size, log_dir, epoch_num, critic_updates):
    return train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates)


if __name__ == '__main__':
    train_command()