# Sentiment Analysis Twitter Dataset

This dataset contains sentiment analysis of tweets regarding various companies on Twitter. Given the message and the company, the task is to evaluate the sentiment of the message about the company. There are three classes in this dataset: positive, negative, and neutral. Messages that are not related to the company (i.e., off-topic) are considered neutral.

## Date of Changes
- 02.29.2024 Added project description
- 04.16.2024 Updated project structure

## Description
- **Task Formulation:** Based on the text of the comment, determine the author's sentiment towards a specific company.
- **Data Source:** [Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- **Data Features:** Moderate number of companies (32), mainly in the gaming industry. Approximately equal distribution of positive and negative comments (28% vs 30%) in the training dataset.
- **Libraries:** PyTorch will be used, experimenting with neural network layers.
- **Production Pipeline:** User uploads a file with Twitter user comments and receives a file with the sentiment analysis of those comments.
  
## Project Structure
```
.
├── Dockerfile
├── code
│   ├── init.py
│   ├── classes
│   │   ├── Model.py
│   │   ├── TwitterDataset.py
│   │   ├── init.py
│   ├── functions
│   │   ├── init.py
│   │   ├── check_accuracy.py
│   │   └── preprocess_and_vectorize.py
│   ├── infer.py
│   └── train.py
├── main.py
├── model.pth
└── requirements.txt
```

