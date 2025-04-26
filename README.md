Intent Detection Project
Overview
This project aims to develop an Intent Detection system for conversational AI. Two different machine learning models are implemented and compared:

Baseline Model: TF-IDF Vectorization + Logistic Regression Classifier

Advanced Model: Fine-tuned BERT (Bidirectional Encoder Representations from Transformers)

The objective is to predict user intent from text input using supervised learning techniques.

Project Structure
Intent-Detection/
│
├── baseline_model/
│   ├── train_baseline.py
│
├── bert_model/
│   ├── train_bert.py
│
├── data/
│   └── sofmattress_train.csv
│
├── results/
│   └── (saved models, confusion matrices)
│
├── report/
│   └── final_report.md
│
├── README.md
├── requirements.txt
Setup Instructions
Clone or download this repository.

Install the required Python libraries:

pip install -r requirements.txt
The following libraries are used:

pandas

scikit-learn

matplotlib

seaborn

nltk

torch (>=2.2)

transformers (>=4.30)

accelerate

Running the Models
1. Train the Baseline Model
Navigate to the baseline model directory and run the script:

cd baseline_model
python train_baseline.py
This will:

Train a Logistic Regression classifier using TF-IDF features.

Save the trained model, vectorizer, and label encoder in the results/ directory.

Generate a confusion matrix plot.

2. Fine-tune the BERT Model
Navigate to the BERT model directory and run the script:

cd bert_model
python train_bert.py
This will:

Fine-tune a pre-trained bert-base-uncased model for the intent classification task.

Save the fine-tuned model and tokenizer in the results/bert_model/ directory.

Note: Fine-tuning BERT may take additional time, especially on CPUs. Using a GPU is recommended if available.

Results
The performance of the two models is compared using evaluation metrics such as accuracy, F1-score, and confusion matrix visualization. Details and analysis are provided in the report/final_report.md file.

Notes
Ensure that the dataset (sofmattress_train.csv) is placed inside the data/ directory.

Adjust batch size and number of epochs in train_bert.py if facing memory constraints.

Model files and outputs are automatically saved in the results/ directory after training.

