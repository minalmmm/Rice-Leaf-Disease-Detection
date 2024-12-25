# Rice Leaf Disease Detection

This project aims to develop a machine learning model to detect diseases in rice leaves. The model uses a Convolutional Neural Network (CNN) architecture to classify rice leaf images into different disease categories, such as healthy and various types of rice diseases.

## Project Pipeline

### 1. **Data Collection**
   - **Objective**: Collect rice leaf images that include healthy leaves and various diseased leaves.
   - **Data Sources**: Public datasets or custom datasets captured from local farms.
   - **Data Format**: Image files in `.jpg`, `.jpeg`, or `.png` formats.

### 2. **Data Preprocessing**
   - **Image Rescaling**: Resize all images to a standard dimension (e.g., 150x150 pixels).
   - **Normalization**: Rescale pixel values to a range of 0 to 1 for better model performance.
   - **Data Augmentation**: Apply transformations (rotation, flipping, zoom, etc.) to artificially increase the size of the dataset and prevent overfitting.

### 3. **Data Splitting**
   - **Training Set**: 80% of the data is used for training the model.
   - **Test Set**: 20% of the data is used to evaluate the model's performance.

### 4. **Model Building**
   - **Model Architecture**: The model uses a CNN architecture, which includes convolutional layers, max-pooling layers, and fully connected layers.
   - **Activation Functions**: ReLU activation is used for hidden layers, and softmax is used for the output layer for multi-class classification.

### 5. **Model Training**
   - **Optimizer**: Adam optimizer is used to minimize the loss function.
   - **Loss Function**: Categorical cross-entropy is used for multi-class classification.
   - **Metrics**: Accuracy is the primary metric for model evaluation.
   - **Early Stopping**: Early stopping is implemented to avoid overfitting by monitoring the validation loss.

### 6. **Model Evaluation**
   - **Accuracy**: The model is evaluated based on its accuracy on the test set.
   - **Confusion Matrix**: A confusion matrix is used to understand the model's performance on each class.
   - **Classification Report**: A classification report is generated, showing metrics like precision, recall, and F1-score for each class.

### 7. **Model Testing**
   - **Testing**: The model is tested on new, unseen rice leaf images to validate its generalization capabilities.
   - **Prediction**: The model provides predictions for new images, classifying them into healthy or diseased categories.

### 8. **Model Deployment**
   - **Web Application**: The trained model is deployed using a web framework like Flask or Streamlit, allowing users to upload images of rice leaves for classification.
   - **Hosting**: The web application is hosted on a cloud platform like Heroku or AWS.

### 9. **Model Maintenance**
   - **Retraining**: The model is periodically retrained with new data to improve its performance.
   - **Monitoring**: The performance of the deployed model is regularly monitored to ensure it remains effective.

## Tools and Technologies Used
- **Programming Language**: Python
- **Deep Learning Libraries**: TensorFlow, Keras
- **Data Manipulation**: NumPy, Pandas
- **Image Processing**: OpenCV, Pillow
- **Web Framework**: Flask or Streamlit (for deployment)

## Output

Below is an example of the output of the Rice Leaf Disease Detection model, showcasing the classification result for a rice leaf image:

![Rice Leaf Disease Detection Example](https://github.com/PankajDevikar/rice-leaf-disease-detection/blob/main/images/img1.jpeg)
![Rice Leaf Disease Detection Example](https://github.com/PankajDevikar/rice-leaf-disease-detection/blob/main/images/img2.jpeg)
![Rice Leaf Disease Detection Example](https://github.com/PankajDevikar/rice-leaf-disease-detection/blob/main/images/img3.jpeg)

This image shows a sample result of the disease detection on a rice leaf, which can help farmers identify the health of their crops quickly.

## Conclusion

The **Rice Leaf Disease Detection** project helps farmers and agricultural specialists by automating the process of diagnosing rice plant diseases. The CNN-based model trained on rice leaf images is an efficient way to monitor rice health and prevent large-scale crop damage. The system is deployed as a web application, making it easily accessible to users who want to upload rice leaf images and get instant predictions.

