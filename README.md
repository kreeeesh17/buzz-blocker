
# SMS Spam Detection App 

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-007ACC?logo=nltk&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![GitHub](https://img.shields.io/badge/Hosted_on-GitHub-black?logo=github)

A machine learning-powered web application that classifies SMS messages as spam or not using NLP techniques and the Multinomial Naive Bayes algorithm. This project includes full model training, evaluation, and a user-friendly Streamlit interface.

---

## 📂 Dataset
- **Source**: [Kaggle - UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Description**: A set of SMS labeled messages as spam or not.

---

## ⚙️ Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Text tokenization using **NLTK**
- Vectorization using **TF-IDF**
- Model comparison using multiple classifiers
- Final model: **Multinomial Naive Bayes**
- Evaluation metrics: **Accuracy, Precision, Confusion Matrix**
- **Streamlit web app** for user interaction

---


Preview of the app can be accessed from [here](https://buzz-blocker.streamlit.app/)

---

## 📁 Project Structure

```
📦 buzz-blocker/
├── app.py                  # Streamlit app
├── model.pkl               # Trained Naive Bayes model
├── vectorizer.pkl          # TF-IDF vectorizer
├── spam.csv                # Original dataset
├── spam_utf8.csv           # UTF-8 converted dataset
├── spam-detection.ipynb    # Training and EDA notebook
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT open-source license
└── README.md               # Contains basic info about the project
```

---

## 🧠 Model Insights

- The dataset was vectorized using TF-IDF to capture term importance.
- Multiple classifiers were tested (e.g. Logistic Regression, SVM).
- **Multinomial Naive Bayes** gave the best results on precision and accuracy.
- The model was saved as `model.pkl` and used directly in the app.

---

## 🛠 Tech Stack

- Python, Pandas, Scikit-learn, NLTK
- TF-IDF Vectorizer
- Streamlit (for frontend)

## 📄 License
This project is licensed under the MIT License.

---


## 📝 Author
Kreesh Modi | IIT Kharagpur Mechanical Engineering

Email: [kreeshmodi2018@gmail.com]
