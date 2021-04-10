import numpy as np
from PIL import Image
from PIL import ImageEnhance
import click
import cv2

from deblurgan.model import generator_model
from deblurgan.utils import load_images, deprocess_image


def test(batch_size):
    #data = load_images('/content/drive/MyDrive/deblur-gan/images/test/last-650-test', batch_size)
    data = load_images('/content/drive/MyDrive/deblur-gan/Biyani/Separate_Dataset/Rotational_blur_Dataset', batch_size)
    #data = load_images('./images/test', batch_size)
    y_test, x_test = data['B'], data['A']
    g = generator_model()
    #g.load_weights('/content/drive/MyDrive/deblur-gan/weights/44/generator_30_0.h5')
    # list = [ 0, 1 , 2 ,3 , 4, 5 , 6, 7 , 8 , 9, 10 ,11 ,12 ,13 ,14 ,15 ,16]
    # for j in range(17):
    #   stringy = '/content/drive/MyDrive/deblur-gan/weights/Biyani_tanh/44/generator_' + str(j) + '_0.h5'
    #g.load_weights('/content/drive/MyDrive/deblur-gan/weights/Rotational_weights/47/generator_16_0.h5')
    g.load_weights('/content/drive/MyDrive/deblur-gan/weights/Rotational_weights/48/generator_49_0.h5')
    #g.load_weights(stringy)
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        print((img))
        # img = Image.fromarray(img.astype(np.uint8))
        # enhancer = ImageEnhance.Brightness(img)
        # enhanced_img = enhancer.enhance(0.75)
        # enhanced_img.save("/content/drive/MyDrive/deblur-gan/Brightness corrected/"+ str(i) + ".png")

        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('results{}.png'.format(i))
        # output = np.concatenate((y, x), axis=1)
        # im = Image.fromarray(output.astype(np.uint8))
        # im.save('results{}.png'.format(i))


@click.command()
@click.option('--batch_size', default=5, help='Number of images to process')
def test_command(batch_size):
    return test(batch_size)


if __name__ == "__main__":
    test_command()
