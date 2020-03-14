# lstm_multi_modal_UNet
Implementation of UNet, multi-modal U-Net, LSTM multi-modal Unet for BRATS15 dataset with Pytorch}

## Prerequisites
* Python >= 3.6  
* Pytorch >= 1.0.0  
* SimpleITK   
* numpy
* scipy


## LSTM multi-modal U-Net
More detail can be found at our origin paper [LSTM Multi-modal UNet for brain tumor segmentation](https://ieeexplore.ieee.org/document/8981027)

You will also find my presentation [slides](https://www.ics.uci.edu/~haoyum3/papers/slides_icivc.pdf) helpful! 


<br>
<img src="https://github.com/HowieMa/lstm_multi_modal_UNet/blob/master/img/model.png" />
<br>


## Train
Before training, please remember to change the 
`data_dir`

For each folder, to train the model, simply run 
`python main.py `

## Citation
If you find this project useful for your research, please use the following BibTeX entry.         

	@inproceedings{xu2019lstm,                  
	  title={LSTM Multi-modal UNet for Brain Tumor Segmentation},                 
	  author={Xu, Fan and Ma, Haoyu and Sun, Junxiao and Wu, Rui and Liu, Xu and Kong, Youyong},            
	  booktitle={2019 IEEE 4th International Conference on Image, Vision and Computing (ICIVC)},               
	  pages={236--240},                  
	  year={2019},                
	  organization={IEEE}               
	}                      


## Reference
[IVD-Net](https://github.com/josedolz/IVD-Net)




