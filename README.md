# ship_detection

This project is based on the [**Airbus Ship Detection Challenge**](https://www.kaggle.com/c/airbus-ship-detection) on Kaggle.

<br/>

The dataset used in this project is provided directly by Airship company, which you can download from [here](https://www.kaggle.com/c/airbus-ship-detection/data) or using Kaggle API by run this command in the terminal:

```cmd
kaggle competitions download -c airbus-ship-detection
```

<br/>

To run the project yourself, you need to set up [Detectron2](https://github.com/facebookresearch/detectron2) environment. You can find official Dockerfile [here](https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile). However, I suggest custom the Dockerfile yourself regarding to you computing device and the driver version. For those who hasn't tried [docker](https://www.docker.com/) before, it's a useful tool to generate container, a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another. To make full use of your computing resource, I also recommend using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) rather than the original version for the better support to NVidia APIs.

<br/>

#### First Step

The first step to run the project is data processing. Scripts in coco/ are for converting the original dataset into [COCO](https://cocodataset.org/#home) format. First to run delete_empty.py to remove images that don't contain a ship. Then to run csv_show_RLE.py to display dataset in RLE style (the original format provided by Kaggle). Then you can generate the COCO annotation by running to_coco.py. Finally, you can check your generated json annotation file by running coco_demo.ipynb. I have provided the final json file in [data_annotations/](data_annotations/). 

<br/>

The weight for the best model can be downloaded [here](https://uchicagoedu-my.sharepoint.com/:u:/g/personal/jingxuanw_uchicago_edu/ETWpCfU05oNPpFNp2l4JIjYB2gMCphSc4x9NgUr7KoNUDg?e=eHyske).