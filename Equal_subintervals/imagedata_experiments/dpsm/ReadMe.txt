Below we explain what each file in this directory is.

1. 'celeba_mean'
First scale CelebA so that values fall between 0 and 1.
Then 'celeba_mean' is a saved torch tensor that contains the mean of the TRAINING data from the CelebA dataset. That is, we have n celebrity prictures, each of them presented as a 3x64x64 matrix. Together they make a 4 dimensional array [n,3,64,64]. If one takes the mean along the first dimension we get 'celeba_mean' with dimensions [3,64,64].
The same holds for 'cifar_mean' and 'imagenet_mean' 

2. 'unet.py'
This file contains the code of the very first Unet.

3. 'torchdiffeq' 
This directory contains the code for Neural ODE optimization. We only use it as a numerical integrator but we do not use it for optimization. 

4. 'train_paral.py'
This file contains the code which performs the training in the dpsm approach. You can directly run this code for CIFAR-10, but in order to do so for CelebA and ImagNet, you should first get the datasets and upload them to '~/diffusion_data'. The main arguments for this file are the number of steps (the number of scores you wish to model), and the number of itreations per score. 
In order to train the score i, the user should give the command python train_paral.py --block_no i . Once the training of that score is finished, the Unet corresponding to that score will be saved as 'modeli' in disk.
Evidently each score can be run in parallel if one possesses multiple devices.

5. 'test_unet.py'
This file contains the code which performs the SDE/CNF generation and likelihood estimation in the dpsm approach. It should only be run when all the scores set in the training files have been modelled. The following arguments should match the ones used in training: Dataset, number of blocks/steps, iterations, and learning rate. Furthermore, the argument 'args.tr_batchsz' in this file should be the same as the as the batch size used in training. 
