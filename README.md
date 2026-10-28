Practical 8 -- Support Vector Machine (SVM) Classification
Aim
To implement and evaluate Support Vector Machine (SVM) classifiers for
predicting machine failure using the AI4I 2020 Predictive Maintenance
dataset.
Dataset
- File: ai4i2020.csv
- Records: 10,000
- Columns: 14
- Target: Machine failure (0 = no failure, 1 = failure)
- The dataset includes machine type, air/process temperature,
  rotational speed, torque, tool wear, and failure-related fields.
The dataset is imbalanced: the target contains 9,661 no-failure records
and 339 machine-failure records.
Practical Workflow
1. Import the required Python libraries.
2. Load and inspect the CSV dataset.
3. Check dataset dimensions, data types, descriptive statistics,
   missing values, duplicates, and target-class distribution.
4. Separate the target (Machine failure) from the input data.
5. Split the data into training and testing sets using an 80:20 split
   (random_state=42).
6. Preprocess the selected input features:
   - Standardize numerical features using StandardScaler.
   - One-hot encode the categorical Type feature using
     OneHotEncoder.
7. Train and evaluate SVM models with four kernels:
   - Linear
   - RBF
   - Polynomial
   - Sigmoid
8. Evaluate each model using accuracy, precision, recall, F1-score,
   classification reports, and confusion matrices.
9. Compare the kernel results with a metrics table and bar chart.
10. Use GridSearchCV with 5-fold cross-validation to tune Linear and
    RBF SVM models, then evaluate the best model on the test set.
Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
Files in This Repository
- Daksh_B_B3_45_ML_Lab_8.ipynb -- Jupyter Notebook containing the
  implementation.
- ai4i2020.csv -- Dataset used by the notebook.
- README.md -- Project overview and instructions.
How to Run
1. Clone or download this repository.
2. Make sure Python and Jupyter Notebook (or Google Colab) are
   available.
3. Install the required libraries:
   pip install pandas numpy matplotlib seaborn scikit-learn
4. Keep ai4i2020.csv in the same folder as the notebook.
5. Open Daksh_B_B3_45_ML_Lab_8.ipynb and run the cells from top to
   bottom.
If using Google Colab, upload ai4i2020.csv to the session or mount
the folder containing it before running the notebook.

Evaluation
The notebook reports Accuracy, Precision, Recall, and F1-score for each
kernel and for the tuned SVM model. It also generates classification
reports, confusion matrices, and a comparison plot. Run the notebook to
obtain the results for your environment.
Conclusion
This practical demonstrates how SVM kernels can be compared for a
machine-failure classification task, and how preprocessing and
cross-validated hyperparameter tuning can be used to evaluate and
improve a model.
