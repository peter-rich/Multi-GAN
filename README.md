# Multi-GAN
### Zhanfu Yang, Purdue University
This is the codes from my graduation thesis(SUN Yat-Sen University), also parts of the codes of the paper I submitted to the ICML 2019.

Paper: http://zhanfuyang.com/Asymetric-Cycle-Gan.pdf
PPT: http://zhanfuyang.com/thesis-ppt.pdf

## Train
`bash train.sh`

## Test
`bash test.sh`

## some parameters
In option.py

`--batch_size ` The batch size

`--save_epoch_freq`  How many epoch to save the parameters of the model

`--which_model_netG ` Use which model for the Generation, Default is `resnet_6blocks`

`--dataroot` The directory of the datasets

`--name` The name of the model 

`--model` Use which model to train the datasets, default is the cycle_gan 

`--no_dropout` 

`--batchSize` Batch Sizes 

`--display_id` Which DEVICE of the GPU to use
