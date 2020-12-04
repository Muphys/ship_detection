# ship_detection

**Jingxuan Wen**

This project is based on the [**Airbus Ship Detection Challenge**](https://www.kaggle.com/c/airbus-ship-detection) on Kaggle.

<br/>

The dataset used in this project is provided directly by Airship company, which you can download from [here](https://www.kaggle.com/c/airbus-ship-detection/data) or using Kaggle API by run this command in the terminal:

```cmd
kaggle competitions download -c airbus-ship-detection
```

<br/>

To run the project yourself, you need to set up [Detectron2](https://github.com/facebookresearch/detectron2) environment. You can find official Dockerfile [here](https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile). However, I suggest custom the Dockerfile yourself regarding to you computing device and the driver version. For those who hasn't tried [docker](https://www.docker.com/) before, it's a useful tool to generate container, a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another. To make full use of your computing resource, I also recommend using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) rather than the original version for the better support to NVidia APIs.

<br/>

### First Step

The first step to run the project is data processing. Scripts in coco/ are for converting the original dataset into [COCO](https://cocodataset.org/#home) format. First to run delete_empty.py to remove images that don't contain a ship. Then to run csv_show_RLE.py to display dataset in RLE style (the original format provided by Kaggle). Then you can generate the COCO annotation by running to_coco.py. Finally, you can check your generated json annotation file by running coco_demo.ipynb. I have provided the final json file in [data_annotations/](data_annotations/). 

<br/>

### Second Step

If you have installed all dependencies mentioned above, then we are ready to construct and train the model. I provide a sample training script in train/ and you can test it by run:

```cmd
python train/train_net.py
```

You can download the initial weight [here](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) and custom the training configuration in the script yourself. By default, the model will be trained with augmented data and the augmenting methods are appointed as rotate, flip, change brightness and contrast. You can implement your onw augmenting methods and add them to the list or not use data augmentation at all.

As mentioned in the report, I tested multiple configurations in the training phase and compared them with each other. The training log for each configuration I tested is also in train/ along with the visualization of it. You can compare them yourself if you wish.

The training process can be time consuming. It may take days for the model to enter a good stage. If you are going to train the model yourself, remember to save the model periodically and enable the logger for future analysis.

The weight for the best model I trained can be downloaded [here](https://uchicagoedu-my.sharepoint.com/:u:/g/personal/jingxuanw_uchicago_edu/ETWpCfU05oNPpFNp2l4JIjYB2gMCphSc4x9NgUr7KoNUDg?e=eHyske).

<br/>

### Third Step

After training, you can visualize the training process using tools/vis.py or tools/vis_simple.py. Also, you can test the final model by running analysis/test.py. This script will randomly choose samples from the dataset and save the input, output and the ground truth in analysis/figs. The analysis/variance.py is specifically for showing the effect of data augmentation on the loss_mask of the model.

<br>

**Please refer to my [project report](https://github.com/Muphys/ship_detection/blob/main/Project_Report.pdf) for more details about the project.**