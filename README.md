# DRAEM-Tensorflow
Tensorflow Implementation of [DRAEM](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf) - ICCV2021:

```
@InProceedings{Zavrtanik_2021_ICCV,
    author    = {Zavrtanik, Vitjan and Kristan, Matej and Skocaj, Danijel},
    title     = {DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {8330-8339}
}
```

A discriminatively trained reconstruction embedding for surface anomaly detection.
DRÆM (Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection) is a method for detecting anomalies in surfaces, 
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
Pass the folder containing the training dataset to the Train_model_1.py script as the --data_path argument and the folder locating the anomaly source images as the --anomaly_source_path argument. The training script also requires learning rate (--lr), epochs (--epochs), path to store checkpoints (--checkpoint_path) and (--object_name) (--load_epoch)  Provide if Reconstructive Model was previously Trained and Training needs to be continued. Default is 0 , Training Starts from zero.
. Example:

```
python Train_model_1.py --object_name 'bottle' --lr 0.0001  --epochs 700 --load_epoch 100 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ 
```
 After 50 epochs the Model is saved in checkpoints_path.
 
 After Reconstructive Model is Trained Next step is to Train Discriminative Model the Discriminative Model automatically laods the latest trained Reconstructive Model from checkpoints_path and loads it.
 (--load_epoch) Provide if Discriminative Model was previously Trained and Training needs to be continued. Default is 0 , Training Starts from zero.
 Example :
 
 ```
 !python model_2.py --data_path ./datasets/mvtec/ --object_namem 'bottle' --anomaly_source_path ./datasets/dtd/images/  --checkpoint_path ./checkpoints/ --load_epoch 100
 ```
 
 # PreTrained Models
 
 For Now only two classes ['Bottle','Carpet'] were trained on a few Images with 100 epochs on both Models. It is recommended to Train it properly but for Inference
 our models can be used.
 PreTrained Models are available [here](https://drive.google.com/file/d/1jP52vmQCJ27jfHNieZD3Bc56vm0Gb9wc/view?usp=share_link)
 We might add more Models in Future
 
 
 # Inference 
 To test the Trained Models use the following script. The script automatically Loads the Latest(highest epochs) Models from checkpoint_path and Displays Images and their respective Predicted Heatmaps.
 Example:
 
  ```
!python test.py --data_path ./datasets/mvtec/  --object_name 'bottle'  --checkpoint_path ./checkpoints/
 ```
 
 # Results
 Both Models were Trained For 100 epochs and only on few Images for Testing Purposes.
 
 ![](https://github.com/hamzakhalil798/DRAEM-Tensoflow/blob/main/images/result_1.PNG)
 ![](https://github.com/hamzakhalil798/DRAEM-Tensoflow/blob/main/images/result_2.PNG)
 
