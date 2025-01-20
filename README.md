<div align="center">
  <a href="https://gdsc.ce.capgemini.com/">
    <img src="images/gdsc_logo.png" alt="Logo" width="600" height="150">
  </a>

<p align="center">
  #GDSC5 Challenge is aimed at using AI to provide a solution that will automate the current time intensive, manual evaluation process involved in the clinical trials of River Blindness disease. Our solution will save years of work and immensely speed up the availability of new treatments to patients afflicted by this illness.
  <br />
</p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#built-with">Built With</a>
    </li>
    <li>
      <a href="#our-approach">Our Approach</a>
      <ul>
        <li><a href="#faster-rcnn">Faster RCNN</a></li>
        <li><a href="#cascade-rcnn">Cascade RCNN</a></li>
      </ul>
    </li>
    <li>
      <a href="#execution-steps">Execution Steps</a>
      <ul>
        <li><a href="#download-preprocess-and-upload-data-to-s3-bucket">Download, preprocess and upload data to s3 Bucket</a></li>
        <li><a href="#training-the-models">Training the models</a></li>
        <li><a href="#model-evaluation">Model Evaluation</a></li>
        <li><a href="#prediction-on-test-dataset-and-ensembling-the-predictions-from-the-two-models">Prediction on test dataset and ensembling the predictions from the two models</a></li>
      </ul>
    </li>
  </ol>
</details><br>

## Built With

* [![Python][Python]][Python-url]
* [![Amazon Sagemaker][Sagemaker]][Sagemaker-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Our Approach

Our final approach is based on model ensembling. We are training two separate pretrained models on our dataset to generate models capable enough to predict valid annotations on a new image. However, to enhance our result quality and make it more robust, we are implementing weighted fusion of these annotations.

### Faster RCNN

Our first model is Faster RCNN ResNet 101. We have created the config for this model which is present in src folder with filename - training_frcnn_5k_r101.py.
Some notable points that we used in the config are as follows:
1. We are using Faster RCNN - ResNet 101 model.
2. The train dataset is repeated 2 times i.e. data is duplicated and augmented.
3. Number of training epochs is set to 24 with learning rate decreasing by a factor of 10 at epoch 12 and 22.
4. Image Scale is set to (5000,5000).
5. Max bounding box predictions per image is set to 400.
6. 90% of data is used for training. Rest 10% is used for validation. For dividing the data into train and validation set, we took 10% of files randomly from each stain for validation set.

### Cascade RCNN

Our second model is Cascade RCNN ResNet 101. We have created the config for this model which is present in src folder with filename - training_crcnn_5k_r101.py.
Some notable points that we used in the config are as follows:
1. We are using Cascade RCNN - ResNet 101 model.
2. Number of training epochs is set to 24 with learning rate decreasing by a factor of 10 at epoch 12 and 22.
3. Image Scale is set to (5000,5000).
4. Max bounding box predictions per image is set to 400.
5. 90% of data is used for training. Rest 10% is used for validation. For dividing the data into train and validation set, we took 10% of files randomly from each stain for validation set.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Execution Steps

### Download, preprocess and upload data to s3 Bucket

All the images, training labels and the test files are present in the s3 bucket - 'gdsc-data-public-us-east-1'. We need to download theses files from the s3 bucket to local sagemaker instance. It was also found out that some of the images were rotated after they were labelled. This orientation information was present in the exif data of the image. We can fix it using exif.
We will use 90% of data for training and rest 10% for validation. For creating the validation set, we will take random 10% of data from each stain. We will then save the train and validation data in src folder.

All the above mentioned steps can be achieved by executing the <u>notebooks/download_and_preprocess_images.ipynb notebook.</u>
Before executing the notebook, we need to set the s3 bucket that we will use for training in src/config.py. We created a s3 bucket named untitled-ipynb and set the value of DEFAULT_BUCKET to untitled-ipynb in src/config.py.

### Training the models

We trained two models - Faster RCNN Resnet 101 and Cascade RCNN ResNet 101. After training the two models, we are ensembling the results of the two models using weighted boxes fusion.
For training we need to follow these steps - 
1. The config for Faster RCNN Resnet 101 model is present at <u>src/training_frcnn_5k_r101.py</u>.
2. Execute the <u>notebooks/training_frcnn_5k_r101.ipynb</u> notebook. It will start a sagemaker training instance and train the model. After model training is completed, execute the last cell of <u>notebooks/training_frcnn_5k_r101.ipynb</u> notebook to download and extract model.
3. Repeat the same steps for Cascade RCNN ResNet 101 model. Its config is present at <u>src/training_crcnn_5k_r101.py</u> and notebook at <u>notebooks/training_crcnn_5k_r101.ipynb</u>.

### Model Evaluation

To evaluate the models, execute the notebook - <u>notebooks/model_evaluation.ipynb</u>. In the notebook, the model predictions are compared with the ground truth data for the validation set. We get a leaderboard score for a range of confidence scores. We can also see the results for individual image and visualize the original and predicted bounding boxes.

### Prediction on test dataset and ensembling the predictions from the two models
<b>
We need to run the <u>notebooks/prediction.ipynb</u> for getting the prediction on test dataset from the two models and ensemble them.
</b><br>
If we already have the predictions from the two models, then we can ensemble them by executing the <u>notebooks/ensemble_results.ipynb</u>.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[Python]: https://img.shields.io/badge/python-111111?style=for-the-badge&logo=python&logoColor=yellow
[Python-url]: https://www.python.org/
[Sagemaker]: https://img.shields.io/badge/Amazon_Sagemaker-DD0031?style=for-the-badge&logo=Sagemaker&logoColor=white
[Sagemaker-url]: https://aws.amazon.com/sagemaker/