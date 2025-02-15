**PCA vs t-SNE Analysis on Wine Quality Dataset**

**1. Overview**

This repository contains Python code to analyze the UCI Wine Quality Dataset using Principal Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE). The purpose is to compare both methods for dimensionality reduction and clustering.

**2. How to Use the Code**

Step 1: Install Required Libraries

Ensure you have the following Python libraries installed:

pip install pandas numpy scikit-learn matplotlib seaborn

Step 2: Dataset Setup

Download the Wine Quality dataset from the UCI Machine Learning Repository:

winequality-red.csv (Red Wine Data)

winequality-white.csv (White Wine Data)

Place these files in the working directory.

Step 3: Running the Code

Execute the Jupyter Notebook (Assignment_2.ipynb) or run the Python script step by step:

# Load and preprocess dataset
python preprocess.py

# Apply PCA
python pca_analysis.py

# Apply t-SNE
python tsne_analysis.py

Step 4: View Outputs

PCA and t-SNE scatter plots will be generated.

The variance explained by PCA components is displayed.

A comparison of PCA and t-SNE is provided.

**3. Project Structure**

 PCA_tSNE_Analysis
├── Assignment_2.ipynb     
├── preprocess.py         
├── pca_analysis.py       
├── tsne_analysis.py     
├── data/                 
│   ├─ winequality-red.csv  
│   ├─ winequality-white.csv
├── outputs/             
└── README.md             

**4. Notes**

PCA is ideal for feature selection and structured dimensionality reduction.

t-SNE is better for visualizing clusters and discovering patterns in high-dimensional data.
