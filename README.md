# Detection of Cyber Attacks in a Network using Machine Learning Techniques

This project demonstrates the end-to-end process of detecting cyber attacks on networks using a set of classical Machine Learning algorithms. The project provides a graphical user interface (GUI) built with `tkinter` in Python, where users can upload a dataset, preprocess it, train various classifiers, and finally test the model on new data to detect potential attacks.

---

## Overview

With an increasing number of cyber attacks, there is a growing need for reliable and efficient detection systems. This project focuses on predicting whether a given network traffic record is an attack or normal using a supervised classification approach. 

**Key Points:**
- Uses the NSL-KDD dataset (or any other dataset of your choice formatted similarly).
- Classification algorithms: SVM, Logistic Regression, Decision Tree, and Random Forest.
- Provides a GUI for ease of use and visualization of results.

---

## Dataset

- **NSL-KDD** (or "clean.txt" in our example) – This dataset is used to detect malicious traffic.
- The dataset has been preprocessed to ensure it contains only numeric features for easy handling by ML algorithms.
- After preprocessing, the dataset is stored in **`clean.txt`**.

---

## Algorithms Used

1. **Support Vector Machine (SVM)**  
   - Uses a kernel-based approach to separate data into different classes.
   - In this project, we use the RBF kernel (`kernel='rbf'`) with class weighting and probability estimation enabled.

2. **Logistic Regression (LR)**  
   - Models the probability that a given input belongs to a certain class using a logistic function.

3. **Decision Tree Classifier (DT)**  
   - A tree-based model that recursively splits data based on feature values to classify the target.

4. **Random Forest Classifier (RFT)**  
   - An ensemble learning method that creates multiple decision trees and merges them to get a more accurate prediction.

Each algorithm's accuracy, confusion matrix, and classification report are displayed, and a plot of the ROC curve is generated.
![Image](https://github.com/user-attachments/assets/8e6e349d-28ba-4e22-88c1-44198fe8d3a3)

---

## UI Flow

A simple tkinter-based GUI is provided with the following buttons:

1. **Upload Dataset** – Choose the dataset file from your local system.  
2. **Preprocess Dataset** – Clean the dataset (remove non-numeric fields, transform labels).  
3. **Generate Training Model** – Split the dataset into train and test sets.  
4. **Run SVM Algorithm**  
5. **Run LR Algorithm**  
6. **Run DT Algorithm**  
7. **Run RFT Algorithm**  
8. **Upload Test Data & Detect Attack** – Load a new test dataset and run the best-trained classifier to detect attacks.  
9. **Accuracy Graph** – Plot a bar graph depicting the accuracy of each classifier used.  

A text area shows the intermediate outputs such as logs, classification reports, and confusion matrices.
![Image](https://github.com/user-attachments/assets/ef2ae3a9-9877-479c-b61b-d4cafa59c722)

---

## Dependencies and Installation

- Python 3.6 or above
- **Required packages:**
  - `numpy`
  - `pandas`
  - `scikit-learn` (`sklearn`)
  - `keras`
  - `matplotlib`
  - `imutils`
  - `tkinter` (default in most Python installations)
  
You can install the required packages with:

```bash
pip install numpy pandas scikit-learn keras matplotlib imutils
```

---

## Running the Project

1. Clone or Download this repository.
2. Make sure the dataset (e.g., KDDTrain+_20Percent.txt or similar) is placed in the NSL-KDD-Dataset folder.
3. Install the dependencies as described above.
4. Execute the IDS.py script:

```bash
python IDS.py
```

The tkinter GUI will open.

---

## Project Walkthrough

### 1. Uploading the Dataset
- Click the "Upload Dataset" button.
- Select the dataset file (e.g., KDDTrain+_20Percent.txt) from your local system.
- The path to the dataset is shown, and a log is printed in the text area indicating "Dataset loaded."

### 2. Preprocessing
- Click "Preprocess Dataset" to convert nominal attributes to numeric form and remove any non-numeric characters.
- A cleaned file named clean.txt is created.
- The text area logs the completed preprocessing step with details of the cleaned data.

### 3. Training the Model
- Click "Generate Training Model" to split the dataset into training and test sets.
- By default, the code uses a 80-20 split.

### 4. Running the Algorithms
Four buttons let you run different algorithms:
- "Run SVM Algorithm"
- "Run LR Algorithm"
- "Run DT Algorithm" (Decision Tree)
- "Run RFT Algorithm" (Random Forest)

Each button:
- Trains the selected algorithm on the training set.
- Performs prediction on the test set.
- Prints accuracy, a confusion matrix, a classification report, and an ROC curve in a new Matplotlib window.

### 5. Detecting Attacks on New Data
- Click "Upload Test Data & Detect Attack".
- Select a test CSV file that has the same number of features (first 38 columns for input features).
- The model (here, Logistic Regression in the provided snippet) makes predictions:
  - Prints "Cyber Attack is Detected!" or "No Attack" for each record in the test data.

### 6. Visualizing Accuracy
- Finally, click "Accuracy Graph" to view a bar chart of the accuracies of all the classifiers in one view.

---

## Results

- **Accuracy**: The text box and Matplotlib plots will display the accuracy of each algorithm.
- **Confusion Matrix**: Understand how many samples from each class were correctly predicted versus misclassified.
- **Classification Report**: Get precision, recall, and F1-score for each class.
- **ROC Curve**: Shows the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR).

This allows you to quickly evaluate which algorithm performs best on detecting cyber attacks.

---

## License

This project is open source and available under the MIT License.
