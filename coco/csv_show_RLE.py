import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import random
np.set_printoptions(threshold=np.inf)   # print all numpy ndarray

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

def csv_show_rle(ImageId, dataset_dir, df):
    img = imread(os.path.join(dataset_dir, ImageId))
    rle_masks = df.loc[df['ImageId'] == ImageId, 'EncodedPixels'].tolist()

    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768))
    for mask in rle_masks:
        binary_mask = rle_decode(mask)
        print('Area: ', np.sum(binary_mask))
        all_masks += binary_mask

    _, axarr = plt.subplots(1, 3)
    axarr[0].axis('off'),
    axarr[1].axis('off'),
    axarr[2].axis('off')
    axarr[0].imshow(img),
    axarr[1].imshow(all_masks),
    axarr[2].imshow(img)
    axarr[2].imshow(all_masks, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    # plt.savefig( os.path.join(ROOT_DIR, '../tmp', 'tmp.png') )
    plt.show()

if __name__ == "__main__":
    dataset_train = '../data/train_v2'
    dataset_test  = '../data/test_v2'
    csv_train =     '../data/train_ship_segmentations_v2.csv'
    
    df = pd.read_csv(csv_train)
    ImageId_list = os.listdir(dataset_train)
    ImageId = random.choice(ImageId_list)
    csv_show_rle(ImageId, dataset_train, df)