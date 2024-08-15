# Fake News Detection System

This repository contains a comprehensive system for detecting fake news using various machine learning algorithms. The system is trained on a dataset of news articles that have been preprocessed and count vectorized. The following algorithms are employed:

## Algorithms

1. **Multinomial Naive Bayes (MNB)**
2. **Passive Aggressive Classifier (PAC)**
3. **Random Forest (RF)**
4. **Logistic Regression (LR)**
5. **XGBoost (XGB)**

## Data Preprocessing

The news articles in the training dataset are preprocessed using standard techniques such as:

- **Removing HTML tags and special characters**
- **Converting to lowercase**
- **Tokenizing into words**
- **Removing stopwords**
- **Lemmatizing words**

The preprocessed text is then count vectorized to create a matrix of token counts.

## Training and Evaluation

The preprocessed and vectorized data is used to train each of the five machine learning algorithms. The models are evaluated using various metrics such as accuracy, precision, recall, and F1-score. The results show that all the models achieve fair accuracies above 95%.

## Usage

To use the fake news detection system, follow these steps:

1. **Clone the repository**
2. **Install the required dependencies**
3. **Load the trained models**
4. **Run 'Streamlit run multi_model.py'**
5. **Select the model for prediction**
6. **Pass a news article to the input feild**
7. **The system will predict whether the news is fake or real based on the trained models**

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvement, please create an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

Citations:
[1] https://github.com/vibhorag101/Fake-News-Detection
[2] https://github.com/kapilsinghnegi/Fake-News-Detection
[3] https://github.com/AtharvaKulkarniIT/Fake-News-Detection-using-Machine-Learning
[4] https://github.com/arindal1/Fake-News-Detector
[5] https://github.com/RishabhSpark/Fake-News-Detection-System



