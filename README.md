# 🚨 Border Intrusion Detection System

An end-to-end **Machine Learning–based Border Intrusion Detection System** that analyzes multi-sensor data to detect and classify intrusion events such as **human, vehicle, animal, false alarm, or normal activity**. The project demonstrates a complete ML workflow — from data preprocessing and model training to deployment using **Streamlit**.

---

## 📌 Project Overview

Border security requires continuous monitoring of large areas using multiple sensors. This project uses **sensor-level data** (motion, sound, thermal, vibration, visibility, and geolocation) to:
* Detect whether an **intrusion has occurred**
* Classify the **type of intrusion**
* Provide an **interactive web interface** for real-time predictions
This repository is designed to showcase **practical machine learning skills** suitable for academic projects and entry-level data science portfolios.

---

## 🧠 Problem Statement

Traditional border monitoring systems often rely on manual surveillance or rule-based alarms, which can lead to:
* High false alarm rates
* Poor scalability
* Delayed response
The goal of this project is to build an **intelligent ML system** that improves detection accuracy by learning patterns from historical sensor data.

---

## 📊 Dataset Description
The dataset consists of sensor readings collected from border monitoring points.

### Features

* `timestamp`
* `sensor_id`
* `latitude`
* `longitude`
* `motion_detected`
* `sound_level_db`
* `thermal_level`
* `vibration_level`
* `visibility`
* `intrusion_type`

### Target Variable

* `intrusion` (Binary: 0 = No Intrusion, 1 = Intrusion)
Intrusion types include:
* Normal
* Human
* Vehicle
* Animal
* False Alarm

---

## ⚙️ Technologies Used

* **Python**
* **Pandas, NumPy** – Data handling
* **Scikit-learn** – Machine Learning models & pipelines
* **Joblib** – Model persistence
* **Streamlit** – Web application
* **Matplotlib / Seaborn** – Data visualization

---

## 🔍 Machine Learning Approach

1. Data Cleaning & Preprocessing

   * Handling categorical variables
   * Feature scaling using `StandardScaler`

2. Model Building

   * Logistic Regression with class balancing
   * Pipeline-based architecture

3. Model Evaluation

   * Accuracy, Precision, Recall
   * Confusion Matrix

4. Deployment

   * Interactive Streamlit application
   * Real-time prediction from user input

---

## 🖥️ Streamlit Application Features

* User-friendly dashboard
* Manual sensor input for prediction
* Real-time intrusion detection result
* Safe imports to prevent crashes in restricted environments

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/border-intrusion-detection.git
cd border-intrusion-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 📈 Results

* Improved intrusion detection compared to rule-based systems
* Reduced false alarms
* Interpretable model coefficients for feature importance

---

## 🚀 Future Improvements

* Use advanced models (Random Forest, XGBoost)
* Real-time sensor data integration
* Alert system with notifications
* Geospatial visualization on maps

---

## 🎯 Purpose of This Project

This project was built to:

* Demonstrate applied machine learning skills
* Showcase end-to-end ML project structure
* Serve as a portfolio project for data science and ML roles

---

## 📜 License
This project is for educational and learning purposes.

---

⭐ *If you find this project useful, feel free to star the repository!*
