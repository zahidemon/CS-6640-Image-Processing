# Project 3: Denoising with Neural Nets

This project will implement a denoising method using neural nets.

**IN YOUR SUBMISSION DO NOT** upload the entire training and testing datasets, instead only include your demo set of images. 

## Parts

0. Download the starting dataset [here,](https://www.dropbox.com/s/jzs3jxr0e3ae9ix/data.zip?dl=0) and starting code [here](.). This starter code is  just a starting point and can be edited as much as you want. 

1. Generate training and testing datasets by completing the `add_noise` function in `create_noisy.py`. Run the script to generate the noisy image datasets. 

2. Complete both `train_loop` and `test_loop` functions in the `train.py`.
    - We've implemented the `NoiseDataset` in `utils.py` for you to use.

3. Define your models.
    - Define a model with one `Conv2d` filter with one input channel and output channel.
    - Define a model with five `Conv2d` filters, with input channel size of first filter as one and output channel size of last filter as one. 
      - All other intermediate channels you can change as you see fit. 
      - Add `BatchNorm2d` layers between each convolution layer for faster convergence.
    - Define a model adding nonlinear activation in between convolution layers from the previous model. 

4. Define hyperparameters (learning rate, weight decay, number of epochs, etc) and optimizer. Experiment with these to try to get better accuracy.

5. For each model architecture, make:
   - plots of training loss and testing accuracy.
   - noisy, denoised, and original images (in one plot). 

6. A demo option, showing denoising results on a subset of testing images. Include four demo setups showing off:
    - all three model architectures.
    - your best performing model, trained on the 'cats' data, denoising the 'pokemon' data.

## Code Submission Guidelines and Tips

**DO NOT** upload the entire training and testing datasets, instead only include your demo set of images. 

Submit the code as a `.zip` file that contains
- A pdf report, explaining what you did as well as your results (see the syllabus for more info).
- All the code, as required above, with comments.
- A folder called `models` with all the pre-trained models.
- A few images (~10) for your demo runs. 
- a `run.sh` script that runs your code, demoing your model using the pretrained models and demo images. It will probably look something like this:
    ```bash
    #!/bin/bash

    python train.py --demo
    ```

**DO NOT** submit code that requires commenting/uncommenting to run various parts, this makes grading very difficult. Instead break code up into functions and/or use the [`argparse`](https://docs.python.org/3/library/argparse.html) library to set options at runtime. 

You're not being graded on code neatness, but that's not an excuse to not use good programming practices - writing reusable, readable code. 
