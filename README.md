
# Digital Paleography of the Ávila Bible: Classification of handwriting in a medieval manuscript


#### -- Project Status: [Completed]

## Objective
The goal of this project was to investigate the Ávila Bible handwriting data set while practising the skills needed to implement a data science project from start to finish.

### Methods Used
* Exploratory Data Analysis
* Preprocessing Pipelines
* Cross Validation (k-folds)
* Machine Learning
* Evaluation Metrics 

### Technologies and Packages
* Python 3.7.3
* numpy 1.17.1
* pandas 0.25.0
* matplotlib 3.1.1
* seaborn 0.9.0
* sklearn 0.21.3

## Project Description
The Ávila Bible is a medieval manuscript that was created in the 12th-century in both Italy and Spain by 12 copyists and is therefore a mix of different styles of writing and decoration. Researchers at the University of Cassino and Southern Lazio in Italy proposed using machine learning to better identify and understand portions of the manuscript. This is a classification problem with the goal of assigning a sample of text to one of the copyists according to the writing style. To that end, the researchers created a data set of over 20,000 data points by deriving features from 800 high-definition images of the pages of the manuscript. The choice of features generated from the images was informed by the expertise of paleographers who work with similar manuscripts. The data set is published on the UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/Avila.


## Repository Organization and Getting Started

├── data/
├── figures/
├── results/
├── reports/
├── src/
├── LICENSE
└── README.md

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw and preprocessed data is found [here](avila/data).
3. Preprocessing,  scripts are being kept [here](avila/src)
4. etc...

