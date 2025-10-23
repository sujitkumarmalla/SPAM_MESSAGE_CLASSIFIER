📧 Spam Mail Detection Web Application

Project Overview

This project implements a real-time Spam Mail Detector deployed as a web application using Streamlit. It allows users to input any email or SMS text and instantly classifies it as either "Spam" or "Ham" (Not Spam).

The classification model relies on a Tf-idf Vectorizer for text-to-numerical feature extraction, followed by a Logistic Regression classifier (or similar simple, effective classification algorithm). The entire model pipeline is saved and loaded for fast, in-browser predictions.

🛠️ Technology Stack

Component

Technology

Purpose

Frontend/Deployment

Streamlit

Used to build the interactive web interface.

Vectorization

Tf-idf Vectorizer (scikit-learn)

Converts raw text into a matrix of weighted token frequencies.

Model

Logistic Regression (scikit-learn),Random ForestClassifier

The core machine learning algorithm for binary classification.

Data Processing,NLP

Python, Pandas, NumPy

Handles data cleaning, preprocessing, and numerical operations.

Model Persistence

pickle

Used to save and load the trained model pipeline (pipe.pkl).

Key Features

Instant Classification: Classifies input text as Spam or Ham immediately upon submission.

NLP Pipeline: Demonstrates a complete Natural Language Processing pipeline, from text input to vectorization and final prediction.

Simple UI: Provides a clean, single-page interface powered by Streamlit for maximum ease of use.

Efficiency: Utilizes a pre-trained and cached model for quick prediction response times.

⚙️ Setup and Installation

Prerequisites

You need Python 3.x installed on your system.

Installation Steps

Clone the Repository (or setup files):

git clone <your_spam_detector_repo_url>
cd spam-mail-detector


Install Required Libraries:
Open your terminal or command prompt in the project directory and run:

pip install streamlit scikit-learn numpy pandas


Ensure Model File is Present:
The application requires the saved model pipeline and TF-IDF vectorizer file, typically named pipe.pkl (or similar), to be located in the same directory as app.py.

▶️ How to Run the App

Navigate to the project directory in your terminal.

Run the Streamlit application:

streamlit run app.py


The app will automatically open in your default web browser (e.g., http://localhost:8501).

🧱 Application Structure

File

Description

app.py

The main Streamlit script. It handles the user interface, text input, model loading, text preprocessing, and displaying the final prediction.

vectorizer.pkl

The serialized machine learning pipeline. This file contains the fitted Tf-idf Vectorizer and the Logistic Regression model.

spam_dataset.csv (Optional)

The dataset used to train the model (if included for reference).

Future Enhancements

Performance Metrics: Display model accuracy, precision, and recall scores within the app sidebar.

Data Preprocessing: Add advanced text cleaning steps like stemming or lemmatization before vectorization.

Model Options: Allow users to switch between different models (e.g., Naive Bayes or SVM) if multiple are trained and saved.
