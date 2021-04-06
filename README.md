# DUGAN
Implementation of Denoising UAV GAN using Keras
Hello
Flow Of Work:

Extraction of Clean Datset from the source
Adding various UAV Imagery Noises to it and saving it to 'A' folder
The noises added were :
  -Gaussian Noise
  -Salt And Pepper Noise
  -Periodic Noise
  -Rotatonal Blur
  -Adaptive Blur
  -Motion Blur
  -Blur
Similarly the Test Dataset also needs to be createdr

Repository Structure:

images:
  -train
  -test
  -scripts
    -train.py
    -test.py
    -organize_your_dataset [to be used only if you have structure of a readily available dataset similar to that of GoPro dataset]
  -deblurgan
    -layer_utils.py
    -losses.py
    -model.py
    -utils.py
  model_architecture
    -g_model.png
    -d_model.png
    -d_on_g_model.png
  weights : contain some pretrained weights on this architecture.
   -They have been arranged on the basis of the dataset used
   - Each contain a .txt file explaining the weights

log.txt : containg mean of d_losses and d_on_g_losses for each epoch ( you can change the printing frequency in the code)
    
  -Git clone this repo 
  -then copy it on your drive so that the weights and everything dont get lost when runtime gets disconnected
  -Run the .ipynb file and save the weights 
  -you can test your weights on test dataset 
  -additional codes for getting PSNR and SSIM for your testing datasets have been given in the .ipynb file
