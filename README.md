# Image Style Transfer by Gatys et al.
Implementation of "Image Style Transfer Using Convolutional Neural Networks, CVPR 2016, by Gatys et al." in PyTorch.
We use pretrained [VGG19 model](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py) from torchvision as our image features extractor, and [L-BFGS](https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html) as our default optimizer.

## Requirements
* python (>= 3.5)
* numpy (>= 1.12)
* pytorch (>= 1.0.0)
* torchvision (>= 0.2.0)

## Usage
Simple evaluation of Image Style Transfer on input `CONTENT` and `STYLE` images can be done as follows

    usage: style_transfer.py [-h] --content CONTENT --style STYLE
                         [--max_size MAX_SIZE] [--total_step TOTAL_STEP]
                         [--sample_step SAMPLE_STEP]
                         [--content_weight CONTENT_WEIGHT]
                         [--style_weight STYLE_WEIGHT]
                         [--total_variance_weight TOTAL_VARIANCE_WEIGHT]
                         [--lr LR]

    Image Style Transfer Using Convolutional Neural Networks, CVPR 2016, by Gatys
    et al.

    optional arguments:
      -h, --help            show this help message and exit
      --content CONTENT     path to content image
      --style STYLE         path to style image
      --max_size MAX_SIZE   rescales image to maximum width or height, default=480
      --total_step TOTAL_STEP
                            total no. of iterations of the algorithm, default=30
      --sample_step SAMPLE_STEP
                            save generated image after every sample step,
                            default=5
      --content_weight CONTENT_WEIGHT
                            content loss hyperparameter, default=1
      --style_weight STYLE_WEIGHT
                            style loss hyperparameter, default=100
      --total_variance_weight TOTAL_VARIANCE_WEIGHT
                            total variance loss hyperparameter, default=0.01
      --lr LR               learning rate for L-BFGS, default=1


If you want to use a different GPU (by default, GPU#0 being used, if available), you can specify it as 
    
    CUDA_VISIBLE_DEVICES=2 python style_transfer.py # uses GPU#2 instead of GPU#0
    
## Results
Some style transfer results on the famous Tuebingen Neckarfront image, each of these optimizations taking 5 mins on average to generate the results. 

### SET A
<p align="center">
    <img width=400 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/contents/tuebingen.jpg"><br>
    <img width=400 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/styles/starry.jpg"> <img width=400 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/tuebingen_starry.png"><br>
    <img width=400 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/styles/scream.jpg"> <img width=400 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/tuebingen_scream.png"><br>
    <img width=400 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/styles/femme.jpg"> <img width=400 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/tuebingen_femme.png"><br>
    <img width=400 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/styles/shipwreck.jpg"> <img width=400 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/tuebingen_shipwreck.png"><br>
    <img width=400 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/styles/night.jpg"> <img width=400 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/tuebingen_night.png"><br>
</p>    

### SET B
<p align="center">
    <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/contents/bradpitt.jpg"> <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/contents/scarlet.jpg"><br>
    <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/styles/fire.jpg"> <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/bradpitt_fire.png"> <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/scarlet_fire.png"><br>
    <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/styles/portrait_0.jpg"> <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/bradpitt_portrait_0.png"> <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/scarlet_portrait_0.png"><br>
    <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/styles/portrait_1.jpg"> <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/bradpitt_portrait_1.png"> <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/scarlet_portrait_1.png"><br>
    <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/styles/whitewalker.jpg"> <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/bradpitt_whitewalker.png"> <img width=250 height=300 src="https://github.com/atulshanbhag/Image-Style-Transfer-Gatys-et-al.-/blob/master/results/scarlet_whitewalker.png"><br>
</p>   

## References
* Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "Image style transfer using convolutional neural networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2414-2423. 2016.
* Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
* Paszke, Adam, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. "Automatic differentiation in pytorch." (2017).
* Liu, Dong C., and Jorge Nocedal. "On the limited memory BFGS method for large scale optimization." Mathematical programming 45, no. 1-3 (1989): 503-528.
