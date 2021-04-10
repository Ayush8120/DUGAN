---

## <mark>**<u>Implementation of Denoising UAV GAN using Keras</u>**</mark>

### Paper Implemented:



[(PDF) An Effective Image Denoising Method for UAV Images via Improved Generative Adversarial Networks](https://www.researchgate.net/publication/325927524_An_Effective_Image_Denoising_Method_for_UAV_Images_via_Improved_Generative_Adversarial_Networks)

> Ruihua Wang,Xiongwu Xiao, Bingxuan Guo,Qianqing Qin, and Ruizhi Chen



## Model Architecture Used:

![](C:\Users\Ayush%20Agrawal\AppData\Roaming\marktext\images\2021-04-10-16-01-16-image.png)



#### Explaination of Generator Model:

<img src="file:///C:/Users/Ayush%20Agrawal/AppData/Roaming/marktext/images/2021-04-10-16-13-01-image.png" title="" alt="" width="576">

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

![](C:\Users\Ayush%20Agrawal\AppData\Roaming\marktext\images\2021-04-10-16-17-50-image.png)

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

![](C:\Users\Ayush%20Agrawal\AppData\Roaming\marktext\images\2021-04-10-16-25-53-image.png)

#### Perceptual Reconstruction Loss:

![](C:\Users\Ayush%20Agrawal\AppData\Roaming\marktext\images\2021-04-10-16-27-16-image.png)

> Ouput taken from conv3_3 layer 

#### Wasserstein Distance:

![](C:\Users\Ayush%20Agrawal\AppData\Roaming\marktext\images\2021-04-10-16-37-31-image.png)

> Unlike the paper, we used Wasserstein distance rather than non-saturating generative adversarial loss.

Lga : is chosen as the Wasserstein Distance

#### Per Pixel Euclidian Distance:

![](C:\Users\Ayush%20Agrawal\AppData\Roaming\marktext\images\2021-04-10-16-26-56-image.png)

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

[Biyani – Google Drive](https://drive.google.com/drive/folders/1R0-X3xl6S7HzZ3IIRJa1J6VVUT6QuPbq?usp=sharing) :Extract them to your directory under ./images folder









**![](https://lh3.googleusercontent.com/-epsOEpVqEUgkUiw29JLzlWjA95NFbljtvzqZxoN-mnZKsK8aZwMZVLgiaXVVWHkDAfi921p7-f56lv2PpyP_KGk_IxpeK48eEKGbz7d_GPKG5_RsMzU1i_SWP7EO_-DeAPuSozk)** 

<mark>Target Image</mark>                             <mark>Blurred Input</mark>                   <mark>Deblurred Output from generator</mark>

Sample Output: <u>when tried with salt and pepper noise</u>

#### **<mark><u>Working With this Repository on Colaboratory:</u></mark>**

-     Git clone this repo 

-     Then copy it on your drive so that the weights and everything dont get lost when      runtime gets disconnected 

-     Run the .ipynb file and save the weights 

-     You can test your weights on test dataset 

-     Additional codes for getting PSNR and SSIM for your testing datasets have been           given in the .ipynb file
