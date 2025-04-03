# EMG Classification and Real-Time Prediction

## 📌 Project Overview
This project focuses on classifying Electromyography (EMG) data to distinguish between **"fist"** and **"release"** movements using machine learning models. It includes:
- Data preprocessing and feature engineering.
- Training models such as Logistic Regression, SVM, Random Forest, and Voting Classifier.
- Real-time file monitoring for continuous EMG signal classification.

## 📂 Dataset
- **Source:** EMG data from `../archive/myo_ds_30l_10ol.npz` and `../Lab_1/emg_filename.csv`.
- **Preprocessing:** Labeled data as **"fist"** or **"release"** based on 2-second intervals.

## 🏗 Features
✔ **Data Preprocessing**
   - Removed unnecessary columns.
   - Standardized the data using `StandardScaler`.

✔ **Custom Transformers**
   - `DropColumnsTransformer`: Drops irrelevant features.
   - `ChunkedMedianTransformer`: Applies median aggregation on chunks of data.

✔ **Machine Learning Models**
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest Classifier
   - Voting Classifier (Combining Logistic Regression & Random Forest)

✔ **Real-Time File Monitoring**
   - Reads new data from the CSV file continuously.
   - Predicts movements using the trained Random Forest model.

✔ **Visualization**
   - Histograms of normalized data for "fist" and "release" movements.

---

## 📦 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/HarishSingaravelan/Gesture-classification
cd Gesture-classification
pip install -r requirements.txt
