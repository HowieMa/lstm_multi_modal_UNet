# lstm_multi_modal_UNet
Implementation of UNet, multi-modal U-Net, LSTM multi-modal Unet for BRATS15 dataset with Pytorch}

## Prerequisites
* Python >= 3.6  
* Pytorch >= 1.0.0  
* SimpleITK   
* numpy
* scipy


## LSTM multi-modal U-Net
More detail can be found at [LSTM Multi-modal UNet for brain tumor segmentation](https://ieeexplore.ieee.org/document/8981027)
You will also find my presentation [slides](https://www.ics.uci.edu/~haoyum3/papers/slides_icivc.pdf) helpful! 
If you use this code in your research, please cite it, thank you! 

<br>
<img src="https://github.com/HowieMa/lstm_multi_modal_UNet/blob/master/img/model.png" />
<br>


## Train
Before training, please remember to change the 
`data_dir`

For each folder, to train the model, simply run 
`python main.py `


## Reference
[IVD-Net](https://github.com/josedolz/IVD-Net)




