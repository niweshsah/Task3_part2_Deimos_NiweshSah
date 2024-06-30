### Wisconsin Breast Cancer Dataset

The **Wisconsin Breast Cancer Dataset** is commonly used for binary classification tasks in machine learning. It contains data collected from breast cancer biopsies, which are used to classify tumors as either malignant (cancerous) or benign (non-cancerous).

#### Features:
- **30 Features**: Numerical values representing characteristics of cell nuclei in images.
  - Examples include radius, texture, perimeter, area, smoothness, etc.
- **Target**: Diagnosis (`M` for malignant, `B` for benign).

#### Data Format:
- **Rows**: Each row represents a single sample (patient).
- **Columns**: The first column is an ID (often removed), the second column is the diagnosis, and the rest are features.

### Logistic Regression on the Dataset

**Logistic Regression** is a linear model used for binary classification. It predicts the probability that a given input belongs to a particular category.

#### Key Concepts:
- **Sigmoid Function**: Maps any real-valued number into a value between 0 and 1, representing the probability of the positive class.
  
    $$
    \sigma(z) = \frac{1}{1 + e^{-z}}
    $$

- **Decision Boundary**: A threshold (commonly 0.5) used to classify the inputs as either class 0 or class 1.


### Explanation:
- **Data Preparation**: The dataset is loaded, and the 'id' and 'diagnosis' columns are handled.
- **Model Training**: Logistic regression is trained on the training set.
- **Evaluation**: Accuracy, classification report, and confusion matrix are displayed, showing the performance of the model.

#### Benefits of Logistic Regression:
- **Interpretable**: The model coefficients can indicate the relationship between features and the outcome.
- **Efficient**: Works well with binary classification tasks and large datasets.
- **Probability Output**: Provides probabilities for predictions, useful in many practical applications.

#### Considerations:
- **Linearity**: Assumes a linear relationship between features and the log-odds of the outcome.
- **Feature Scaling**: It’s beneficial to scale features for better performance.

## Running the Script

1. Save the code as breast_cancer_classification.py.
2. Install required libraries: pip install numpy scikit-learn (if not already installed).
3. Run the script from your terminal: python breast_cancer_classification.py

The script will print the accuracy of the model on the testing data.
