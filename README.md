# DRAEM-Tensorflow
A discriminatively trained reconstruction embedding for surface anomaly detection.
This is tensorflow implementation of [DRAEM](https://arxiv.org/pdf/2108.07610v2.pdf).DRÆM (Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection) is a method for detecting anomalies in surfaces, 
such as defects or damage, using a combination of reconstruction and classification techniques.
## Anomaly Detection Process
![](https://github.com/farazBhatti/DRAEM-Tensoflow/blob/main/images/result.png)

#Datasets
To train on the MVtec Anomaly Detection dataset download the data and extract it. The Describable Textures dataset was used as the anomaly source image set in most of the experiments in the paper. You can run the download_dataset.sh script from the project directory to download the MVTec and the DTD datasets to the datasets folder in the project directory:
> ./scripts/download_dataset.sh
