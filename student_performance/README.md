# Student Performance Prediction (ML)

A Machine Learning project built using Python and Scikit-Learn to predict a student's final examination score based on study hours, attendance percentage, and previous academic performance.

## 🚀 Features

- Predicts a student's final score using Multiple Linear Regression.
- Uses study hours, attendance, and past score as input features.
- Splits the dataset into training and testing data.
- Evaluates the model using R² Score and Mean Squared Error (MSE).
- Provides a custom prediction for a new student.

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **Libraries:**
  - NumPy
  - Pandas
  - Scikit-Learn

## 📊 Input Features

| Feature | Description |
|---|---|
| Hours Studied | Number of hours studied |
| Attendance | Attendance percentage |
| Past Score | Student's previous academic score |
| Final Score | Target value to be predicted |

## 🤖 Machine Learning Model

The project uses **Multiple Linear Regression** to predict the final score.

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

The trained model is then evaluated using:

- **R² Score**
- **Mean Squared Error (MSE)**

## 🔮 Sample Prediction

Example student:

- **Hours Studied:** 7
- **Attendance:** 85%
- **Past Score:** 75

The trained model predicts the student's expected final score.

> The exact predicted score and evaluation values are generated when the program is executed.

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPO-LINK>