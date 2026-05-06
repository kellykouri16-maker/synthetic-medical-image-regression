Thesis-related implementation repositories. This project uses TensorFlow/Keras to train a convolutional neural network on synthetic grayscale images containing two elliptical regions. The model predicts the intensity values and spatial center coordinates of the two regions. The pipeline includes synthetic data generation, noise simulation, two-phase training, regression evaluation, residual analysis, and visualization of prediction accuracy.

## Results

### Synthetic image examples
![Synthetic examples](images/synthetic_examples.png)

### Training and validation loss
![Training and validation loss](images/training_validation_loss.png)

### Prediction performance
![Predicted vs true scatter plots](images/predicted_vs_true_scatter.png)

### Mean squared error per output
![MSE per output](images/mse_per_output.png)

### Center distance error distribution
![Center distance error histogram](images/center_distance_error_histogram.png)

