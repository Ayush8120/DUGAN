import keras.backend as K
# from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import VGG19
from keras.models import Model
import numpy as np
import tensorflow as tf


# Note the image_shape must be multiple of patch_shape
image_shape = (320, 320, 3)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def perceptual_loss(y_true, y_pred):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    factor = 1/(320*320*3)
    loss_euc_dist = 0.01*factor*(K.sqrt(K.sum(K.square(y_true - y_pred), keepdims=True)))#2 zero aur lagaye
    loss_perceptual = 0.000005*factor*(K.sqrt(K.sum(K.square(loss_model(y_true) - loss_model(y_pred)), keepdims = True)))  #ye generator ka loss hai
    return (loss_perceptual + loss_euc_dist)

def euc_dist_keras(image_full_batch, generated_image):
    return factor*(K.sqrt(K.sum(K.square(image_full_batch - generated_image),keepdims=True)))   #ye generator ka loss hai

def ns_generator_loss(y_true, y_pred):
    return -tf.math.log(tf.math.sigmoid(y_pred))   #ye discriminator ka loss hai

# class perceptual_loss():
# #y_true,y_pred
# 	def contentFunc(self):
# 		conv_3_3_layer = 14
# 		cnn = models.vgg19(pretrained=True).features
# 		cnn = cnn.cuda()
# 		model = nn.Sequential()
# 		model = model.cuda()
# 		for i,layer in enumerate(list(cnn)):
# 			model.add_module(str(i),layer)
# 			if i == conv_3_3_layer:
# 				break
# 		return model
		
# 	def __init__(self, loss, y_true, y_pred):
# 		self.criterion = loss
# 		self.contentFunc = self.contentFunc()
#     # self.fakeIm = y_pred
#     # self.realIm = y_true
			
# 	def get_loss(self, fakeIm, realIm):
# 		f_fake = self.contentFunc.forward(fakeIm)
# 		f_real = self.contentFunc.forward(realIm)
# 		f_real_no_grad = f_real.detach()
# 		loss_1 = self.criterion(f_fake, f_real_no_grad)
#     loss_euc_dist = 0.0000000325520833*(K.sqrt(K.sum(K.square(y_true - y_pred),keepdims=True)))#2 zero aur lagaye
# 		loss = loss_1 + loss_euc_dist
#     return loss