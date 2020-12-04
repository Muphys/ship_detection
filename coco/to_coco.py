import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import pandas as pd

from skimage.io import imread
import matplotlib.pyplot as plt

dataset_train = '../data/train_v2'
csv_train = 	'../data/train_ship_segmentations_v2.csv'
IMAGE_DIR = dataset_train

df = pd.read_csv(csv_train)

INFO = {
    "description": "Kaggle Dataset",
    "url": "https://github.com/Muphys",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "Jingxuan Wen",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'ship',
        'supercategory': 'ship',
    },
]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# mask_rle(string) --> rle_decode() -->  np.ndarry(np.unit8)    
# shape: (height,width) , 1 - mask, 0 - background
def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts =  np.asarray(s[0::2], dtype=int)
    lengths = np.asarray(s[1::2], dtype=int)

    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def save_bad_ann(image_name, mask, segmentation_id):
    img = imread(os.path.join(IMAGE_DIR, image_name))
    _, axarr = plt.subplots(1, 3)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(mask)
    axarr[2].imshow(img)
    axarr[2].imshow(mask, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    if not os.path.exists('tmp'):
    	os.makedirs('tmp')
    plt.savefig( os.path.join('./tmp', image_name.split('.')[0] +'_' +str(segmentation_id) +'.png') )
    plt.close()

def main():
    # json dictionary
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    for root, _, files in os.walk(IMAGE_DIR):
        image_paths = filter_for_jpeg(root, files)
        num_of_image_files = len(image_paths)

        for image_path in image_paths:
            image = Image.open(image_path)
            image_name = os.path.basename(image_path)
            image_info = pycococreatortools.create_image_info(
                image_id, image_name, image.size)
            coco_output["images"].append(image_info)

            # get maskes
            rle_masks = df.loc[df['ImageId'] == image_name, 'EncodedPixels'].tolist()
            num_of_rle_masks = len(rle_masks)

            for index in range(num_of_rle_masks):
                binary_mask = rle_decode(rle_masks[index])
                class_id = 1    # all images are 1, with ship
                category_info = {'id': class_id, 'is_crowd': 0}
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)

                # filter out low-quality annotations
                # save low-quality annotations for later
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                else:
                    save_bad_ann(image_name, binary_mask, segmentation_id)
                    
                # assign an id to each annotation (no matter if it's filtered out)
                segmentation_id = segmentation_id + 1   

            if not image_id%100: print("%d of %d is done."%(image_id,num_of_image_files))
            image_id = image_id + 1

    with open('../data/annotations/train_ship_instances_v2.json', 'w') as output_json_file:
        # json.dump(coco_output, output_json_file)
        json.dump(coco_output, output_json_file,indent=4)

if __name__ == "__main__":
    main()
