# DRAEM-Tensorflow
A discriminatively trained reconstruction embedding for surface anomaly detection.
This is tensorflow implementation of [DRAEM](https://arxiv.org/pdf/2108.07610v2.pdf).DRÆM (Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection) is a method for detecting anomalies in surfaces, 
such as defects or damage, using a combination of reconstruction and classification techniques.
## Anomaly Detection Process
![](https://github.com/farazBhatti/DRAEM-Tensoflow/blob/main/images/result.png)

# Datasets
To train on the MVtec Anomaly Detection dataset download the data and extract it. The Describable Textures dataset was used as the anomaly source image set in most of the experiments in the paper. You can run the download_dataset.sh script from the project directory to download the MVTec and the DTD datasets to the datasets folder in the project directory:
```
./scripts/download_dataset.sh

```

# Training
DRAEM has two Models. A reconstructive Model that reconstructs the Augmented Image and A Discriminative Model that predicts the Anomaly Mask.
First Train the Reconstructed Modedel by :
Pass the folder containing the training dataset to the Train_model_1.py script as the --data_path argument and the folder locating the anomaly source images as the --anomaly_source_path argument. The training script also requires learning rate (--lr), epochs (--epochs), path to store checkpoints (--checkpoint_path) and (--object_name)
. Example:
```
python Train_model_1.py --object_name 'bottle' --lr 0.0001  --epochs 700 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ 

```
