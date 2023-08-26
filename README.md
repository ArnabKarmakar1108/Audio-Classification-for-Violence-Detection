
# Audio Classification for Violence Detection

This project focuses on the automatic detection of violence-related sounds or behaviors in audio data using machine learning techniques. By leveraging the `librosa` library, audio features such as mel frequency cepstral coefficients (MFCC), chroma features, and mel spectrograms are extracted from the audio recordings. These features capture essential characteristics of the audio signal that are crucial for classification.

## Features Used

1. Mel Frequency Cepstral Coefficients (MFCC): These coefficients represent the spectral features of the audio signal, capturing timbral and perceptual characteristics.
2. Chroma Features: These features provide insights into the tonal content of the audio, making them useful for identifying musical attributes.
3. Mel Spectrogram: This is a visual representation of the spectrum of frequencies in the audio signal over time.

## Methodology

1. Data Collection: Gather a labeled dataset containing both violent and non-violent audio recordings.
2. Feature Extraction: Utilize the `librosa` library to extract MFCC, chroma, and mel spectrogram features from the audio data.
3. Model Building: Develop a Convolutional Neural Network (CNN) model to classify the extracted features. The CNN can learn intricate patterns and relationships in the audio data.
4. Training: Split the dataset into training and validation sets, and train the CNN using the extracted features as input and the corresponding labels as output.
5. Evaluation: Evaluate the trained model's performance using metrics such as accuracy, precision, recall, and F1-score on a separate test dataset.
6. Inference: Deploy the trained model to classify new audio data and detect instances of violence.

## Getting Started

1. Install the necessary dependencies listed in the `requirements.txt` file.
2. Prepare your labeled audio dataset and organize it appropriately.
3. Run the feature extraction script to generate feature matrices from the audio data.
4. Build, train, and evaluate the CNN model using the extracted features and labels.
5. Utilize the trained model for real-time or batch classification of audio data.

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`

## Acknowledgments

- The project utilizes the `librosa` library for audio feature extraction.
- The CNN architecture is inspired by state-of-the-art methods in audio classification.
