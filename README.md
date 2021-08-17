---

## <mark>**<u>Implementation of Denoising UAV GAN using Keras</u>**</mark>

### Paper Implemented:



[(PDF) An Effective Image Denoising Method for UAV Images via Improved Generative Adversarial Networks](https://www.researchgate.net/publication/325927524_An_Effective_Image_Denoising_Method_for_UAV_Images_via_Improved_Generative_Adversarial_Networks)

> Ruihua Wang,Xiongwu Xiao, Bingxuan Guo,Qianqing Qin, and Ruizhi Chen



## Model Architecture Used:

**![](https://lh4.googleusercontent.com/qvOizGJWNjBmkinDzuO2BxIVGd8NmnDD3rDczSylJ8uqeQqW7Fi-wYcnvJYLwPdtw4ceEhc6eOE8NEQEYAYEuPFTrLVPUZmoaB1N0hcEqR-dBjk0l8Az7APkFWlOSJ5XV6_zo1yk)**

#### Explaination of Generator Model:



**![](https://lh4.googleusercontent.com/uC4lEgZFqnTz0g9CNe7ibI4Ic14fh5ZoII9kOFawoZt4U6vDNnI_gJ9Z-uROJyITOkXvzZyYyc_WjSTMgl7HTWEGy1MQLu91Y5pijXvgulftWleMb8xgZNy_M9ozqUx7N3emPqgV)**

> Residual Block: (Conv2D + Batch Normalization[with : ReLu]) x2
> 
>     -C(64)B(r)C(64)B(r)SC : Residual Block with Skip Connection
> 
>     -We have used a skip connection at the end of each residual block
> 
>     - We have added such 14 Residual Blocks
> 
>     -  C(r,64) : set of vonvolutional layers with 64 feature maps and ReLu         Activation
> 
>     - B(r) : batch normalization with activation ReLu
> 
>     - C(t,3) : convolutional layer with 3 feature maps and Tanh Activation



#### Explaination of Discriminator Model:

**![](https://lh6.googleusercontent.com/5julg8IfmJj6XLd9JRcsdlOMJ7jlahgNnBkcEBWJCgCd-ItJCD9fts-BCDgg1r3LS0hf_ApRILARFeDsf37uY9W3y8pbwdxjjqVOSGaa0_N2HNDOL8Q0nNemrzpKfuZ5_Nf_rHVD)**

> lr denotes : Leaky ReLu Activation
> 
> C(128)BN(lr) : set of convolutional layers with 128 filters followed by Batch Normalization with activation function Leaky ReLu
> 
> D refers to the dense layer 

#### Slight Changes made :

> -Removal of SC layer written in discriminator model
> 
> -Addition of another dense layer of 1024 neurons before the Discriminator Sigmoid Layer
> 
> -"Same" padding used in Generator Model 
> 
> -"Same" padding along with variable strides used in different layers



## Loss Function Used :

The usage of pixel loss functions often leads to perpectually unsatisfying solutions with overly smooth textures.

Additionaly, the quality of the image features can be improved by the perceptual reconstruction loss so that it can meet the requirements of UAV images for features acquired from ground objects textures.

Thus here, Eucidian Loss is combined with perceptual reconstrction loss

#### Overall DUGAN Loss:



**![](https://lh6.googleusercontent.com/byHPEauHSWG9VwV6cTJOoin9e7AvGUYdEIY9txkHIidpV-OOEAM280fLpbZbbejf3Ck5EW3ukLBHIVyV-z0r8VQS8Zc4H58YDEaF_P-gXmfG7lvTTe5-WxvkVvhvDllFcxcIArOH)**

#### Perceptual Reconstruction Loss:



**![](https://lh5.googleusercontent.com/XkPa3nOTpNwCUNIlHyjJxSs3hevS7g22yB2vOyOsgFO06TaBhj6MemM04TwPLFAyFLiN0k4_Y7Z_jz-Nq1-kKGUXYRgXLhRDYJk8YjGOteuIzZSIZ-FbM-yAjvckr0bN1KiVVKnU)**

> Ouput taken from conv3_3 layer 

#### Wasserstein Distance:

**![](https://lh5.googleusercontent.com/GNz1cp7o7dfvcaxVJRhnoFFx8_jg1EXeZiKLA2ij2LLYBb8ow_xMwk7NDAnIpnhdaXMWdxJ7E8TRdqz1CSqlUqNYd7lPcA2w1gn4UHShcB8VlOUGsMoXqgwbtJ7eAB-FgOZDgt5c)**

> Unlike the paper, we used Wasserstein distance rather than non-saturating generative adversarial loss.

Lga : is chosen as the Wasserstein Distance

#### Per Pixel Euclidian Distance:

**![](https://lh4.googleusercontent.com/ZUGPm3VzIi713cbGaq3jQ3KutttOJ_-hLAB8VDHXtqfGV94pqrb-w222gA3rGAD5G2dY6iXwM3bCXWXcXmJvmRlqSc96ZXzHQ6l0cf0POiVAvS-A7w3RPXr-ILOXI6Q-e8Ve3qgK)**

x and y can be treated as hyper parameters and thus will vary model to model, depending on our training dataset

 Flow Of Work:

## <u><mark>DATASET CREATION/IMPORTING:</mark></u>

- Extraction of Clean Datset from the source Adding various UAV Imagery Noises to it and saving it to 'A' folder 
  
  -    The noises added were :
    
    ```
     -*Gaussian Noise
    
     -*Salt And Pepper Noise*
    
     -*Periodic Noise*
    
     -*Rotatonal Blur
    
     -*Adaptive Blur*
    
     -*Motion Blur*
    
     -*Blur 
    ```

- Similarly the Test Dataset also needs to be created 

## **<mark><u>Repository Structure</u>:</mark>**

***images***: 

> -train 
> 
> -test 

***scripts***

> -train.py 
> 
> -test.py 
> 
> -organize_your_dataset [to be used only if you have structure of a readily         available dataset similar to that of GoPro dataset] 

***deblurgan***

> -layer_utils.py
> 
> -losses.py
> 
> -model.py
> 
> -utils.py

***model_architecture*** 

> -g_model.png 
> 
> -d_model.png 
> 
> -d_on_g_model.png 

***weights*** : contain some pretrained weights on this architecture. -They have been arranged on the basis of the dataset used for their training.

- Each contain a .txt file explaining the weights

***log.txt :*** containg mean of d_losses and d_on_g_losses for each epoch ( you can change the printing frequency in the code)

### To get the dataset used in this code:

Please access the training and testing datsets used from here:

[Biyani – Google Drive](https://drive.google.com/drive/folders/1XhVh2tKQT1pyuRKJ-HH9yByMLW0ZtJHT) :Extract them to your directory under ./images folder

**![](https://lh3.googleusercontent.com/-epsOEpVqEUgkUiw29JLzlWjA95NFbljtvzqZxoN-mnZKsK8aZwMZVLgiaXVVWHkDAfi921p7-f56lv2PpyP_KGk_IxpeK48eEKGbz7d_GPKG5_RsMzU1i_SWP7EO_-DeAPuSozk)** 

###### <mark><u>*Target Image*</u></mark>                             <mark>*<u>Blurred Input</u>*</mark>                  <mark><u>*Deblurred Output from generator*</u></mark>



Sample Output: <u>when tried with salt and pepper noise</u>

#### **<mark><u>Working With this Repository on Colaboratory:</u></mark>**

-     Git clone this repo 

-     Then copy it on your drive so that the weights and everything dont get lost when      runtime gets disconnected 

-     Run the .ipynb file and save the weights 

-     You can test your weights on test dataset 

-     Additional codes for getting PSNR and SSIM for your testing datasets have been           given in the .ipynb file
