# Heart-Disease-Prediction
# Heart Disease Prediction Using Logistic Regression

## Project Overview
This project aims to predict the likelihood of heart disease in individuals based on various medical and demographic factors using Logistic Regression. The dataset used is the Framingham Heart Study dataset, which contains information about various health metrics and whether or not the individual developed coronary heart disease (CHD) within ten years.

## Dataset
The dataset used in this project is `framingham.csv`, which includes the following columns:
- **age**: Age of the individual
- **sex**: Gender of the individual
- **cp**: Chest pain type
- **trestbps**: Resting blood pressure
- **chol**: Serum cholesterol in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
- **target**: Presence of heart disease (1 = yes, 0 = no)

## Data Preprocessing
1. **Loading the Dataset**: The dataset is loaded using Pandas.
   ```python
   disease_df = pd.read_csv("framingham.csv")
   ```

2. **Handling Missing Values**: Rows with any missing values are dropped.
   ```python
   disease_df.dropna(axis=0, inplace=True)
   ```

3. **Renaming Columns**: The `male` column is renamed to `Sex_male` for clarity.
   ```python
   disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)
   ```

4. **Removing Unnecessary Columns**: The `education` column is dropped as it is not needed for the analysis.
   ```python
   disease_df.drop(['education'], inplace=True, axis=1)
   ```

## Exploratory Data Analysis
- **Distribution of Target Variable**: The target variable `TenYearCHD` is analyzed to check for class imbalance.
  ```python
  print(disease_df.TenYearCHD.value_counts())
  ```

## Model Building
1. **Logistic Regression**: Logistic Regression is used to predict the probability of heart disease based on the input features.
2. **Data Splitting**: The dataset is split into training and testing sets.
3. **Model Training**: The Logistic Regression model is trained on the training data.
4. **Model Evaluation**: The model is evaluated using various metrics such as accuracy, precision, recall, and the confusion matrix.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels

## Installation
To install the required packages, run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

## Usage
1. Clone the repository.
2. Ensure all dependencies are installed.
3. Run the Jupyter Notebook or Python script to preprocess the data and train the model.
4. Evaluate the model's performance.

## License
This project is licensed under the MIT License.

## Acknowledgements
- The Framingham Heart Study dataset is a publicly available dataset used for educational and research purposes.
- Inspiration from various online resources and academic projects related to heart disease prediction.

