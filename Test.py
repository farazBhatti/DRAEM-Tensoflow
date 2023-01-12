import argparse
import tensorflow as tf
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from model import ReconstructiveSubNetwork,DiscriminativeSubNetwork
from dataloader import MVTecDRAEMTestDataset

def predict(object_name:str,checkpoint_path:str, data_path:str):
    model = ReconstructiveSubNetwork((256,256,3))
    model_seg = DiscriminativeSubNetwork((256,256,6))
    #Loading Latest Reconstruction Model
    files = glob.glob(f'{checkpoint_path}model_{object_name}_weights_*.h5')
    latest_file = max(files, key=lambda x: int(re.search(r'_(\d+)\.h5', x).group(1)))

   # Get the epoch number
    epoch = int(re.search(r'_(\d+)\.h5', latest_file).group(1))
    
    model.load_weights(latest_file)
    print(f'checkpoint {epoch} loaded for Reconstruction Model')
    #Loading Discriminative Model
    # Get all .h5 files in the directory that match the pattern
    files = glob.glob(f'{checkpoint_path}model_seg_{object_name}_weights_*.h5')

    # Find the file with the highest epoch number
    latest_file = max(files, key=lambda x: int(re.search(r'_(\d+)\.h5', x).group(1)))

    # Get the epoch number
    epoch = int(re.search(r'_(\d+)\.h5', latest_file).group(1))

    # Load the model
    model_seg.load_weights(latest_file)
    
    print(f'checkpoint {epoch} loaded for  Discriminative Model Loaded')
    dataset = MVTecDRAEMTestDataset(data_path + object_name + "/test/", resize_shape=(256,256))

    for i_batch, sample_batched in enumerate(dataset):
        if i_batch == len(dataset):
            break
        gray_batch = sample_batched["image"]
        gray_batch = np.expand_dims(gray_batch, axis=0)
        plt.imshow(gray_batch[0], cmap='gray') 
        plt.title(f'{object_name}: Original Image')
        plt.show()
    
        gray_rec = model(gray_batch)
        joined_in = tf.concat([gray_rec, gray_batch], axis=3)
        out_mask = model_seg(joined_in)
        plt.imshow(out_mask[0][:,:,0])
        plt.title(f'{object_name}: Anomaly Heatmap')
        
        plt.show()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='path to the data')
    parser.add_argument('--object_name', type=str, required=True, help='name of the object being trained on')
    parser.add_argument('--checkpoint_path', type=str,required=True , help='directory to load checkpoints from')
    args = parser.parse_args()
    predict(args.object_name, args.checkpoint_path, args.data_path)

