import argparse
import tensorflow as tf
from dataloader import MVTecDRAEMTestDataset,MVTecDRAEMTrainDataset
import os
from model import ReconstructiveSubNetwork,DiscriminativeSubNetwork
from loss import FocalLoss
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa

def train(data_path, object_name, anomaly_source_path, lr, epochs, load_epoch, checkpoint_path):
    #Models
    model = ReconstructiveSubNetwork((256,256,3))
    model_seg = DiscriminativeSubNetwork((256,256,6))
    dataset = MVTecDRAEMTrainDataset(data_path + object_name + "/train/good/", anomaly_source_path, resize_shape=(256,256))
    add=0
    # Set up optimizers
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    #Loading Checkpoint
    if os.path.exists(checkpoint_path+f'model_{object_name}_weights_{load_epoch}.h5'):
        model.load_weights(checkpoint_path+f'model_{object_name}_weights_{load_epoch}.h5')
        print(f'Checkpoint {load_epoch} loaded')
        add=load_epoch

    else:
      print('Training Reconstructive Model from 0')
    # Training loop
    for epoch in range(epochs):
        # Get next batch of data
        for i_batch, sample_batched in enumerate(dataset):
            if i_batch ==len(dataset):
                break
            aug_image = sample_batched["augmented_image"]
            anomaly_mask = sample_batched["anomaly_mask"]

            # Preprocess data
            anomaly_mask = np.expand_dims(anomaly_mask, axis=0)
            aug_image = np.expand_dims(aug_image, axis=0)
            gray_batch = sample_batched['image']
            gray_batch = np.expand_dims(gray_batch, axis=0)
            # Run models and calculate losses
            with tf.GradientTape() as tape:
                with tf.device('/GPU:0'):
                    gray_rec = model(aug_image)
                    image=gray_rec.numpy()
                    plt.imshow(image[0])
                    plt.title(f'{object_name}: Reconstructed Image')
                    plt.show()
                    plt.imshow(aug_image[0])
                    plt.title(f'{object_name}: Augmented Image')
                    plt.show()
                    loss_mse=tf.keras.losses.MeanSquaredError()
                    loss_m=loss_mse(gray_batch,gray_rec)

                    ssim = tf.image.ssim(gray_batch,gray_rec, max_val=255, filter_size=11,
                                        filter_sigma=1.5, k1=0.01, k2=0.03)
                    ssim_loss = (1 - ssim)
                    loss = ssim_loss+loss_m

            # Calculate gradients and optimize model
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Print loss values at the end of each epoch
            print('Epoch {}: loss = {}'.format(epoch+1, loss))
        if (epoch % 50)==0:     
            # Save trained models
            model.save_weights(checkpoint_path+f'model_{object_name}_weights_{epoch+add}.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='path to the data')
    parser.add_argument('--object_name', type=str, required=True, help='name of the object being trained on')
    parser.add_argument('--anomaly_source_path', type=str, required=True, help='path to the anomaly source data')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--load_epoch', type=int, default=0, help='which checkpoint to load')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='directory to save checkpoints to')
    args = parser.parse_args()
    train(args.data_path, args.object_name, args.anomaly_source_path,args.lr, args.epochs, args.load_epoch, args.checkpoint_path)
