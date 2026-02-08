# Comparative Analysis of Classification Algorithms

A beginner-friendly comparative analysis of three machine learning classification algorithms on the famous Titanic survival dataset.

## 🎯 Objective

To understand the differences between **linear vs non-linear models** and explore the **bias-variance tradeoff** by comparing:
- **Logistic Regression** (Linear)
- **Decision Tree** (Non-linear)
- **Random Forest** (Ensemble)

## 📊 Dataset

**Titanic Survival Dataset**
- **Training data**: 893 passengers with survival outcomes
- **Test data**: 420 passengers for prediction
- **Target variable**: `Survived` (0 = died, 1 = survived)
- **Key features**: Passenger class, gender, age, fare, family size, etc.

### Key Insights from Data Exploration:
- **Women had ~74% survival rate** vs ~19% for men
- **First class had ~63% survival** vs ~24% for third class
- **Gender and passenger class are strong predictors**

## 🤖 Models Implemented

### 1. Logistic Regression
- **Type**: Linear classifier
- **How it works**: Draws a straight line to separate survivors vs non-survivors
- **Bias-Variance**: Higher bias, lower variance
- **Scaling**: Tested both scaled and unscaled versions

### 2. Decision Tree
- **Type**: Non-linear classifier
- **How it works**: Asks yes/no questions (e.g., "Is passenger female?" or "Is fare > $50?")
- **Bias-Variance**: Lower bias, higher variance
- **Parameter tuning**: Tested different max_depth values (None, 5, 10, 15)

### 3. Random Forest
- **Type**: Ensemble classifier
- **How it works**: Combines many decision trees and averages their votes
- **Bias-Variance**: Balanced approach, reduces variance while maintaining flexibility
- **Parameter tuning**: Tested different numbers of trees (50, 100, 200)

## 📋 Methodology

### 1. Data Preprocessing
- Convert categorical `Sex` to numeric (female=1, male=0)
- Handle missing values using median imputation
- Select key features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`

### 2. Feature Analysis
- **Distribution plots**: Understand feature ranges and scaling needs
- **Contour plots**: Visualize relationships between features and survival
- **Exploratory analysis**: Survival rates by gender and class

### 3. Model Training & Evaluation
- Train all three models with default settings
- Test scaling impact on Logistic Regression
- Perform simple parameter tuning
- Evaluate using: **Accuracy**, **Precision**, **Recall**, **ROC-AUC**

### 4. Comparative Analysis
- Compare scaled vs unscaled Logistic Regression
- Test different Decision Tree depths
- Test different Random Forest tree counts
- Select optimal configurations for final comparison

## 🗂️ File Structure

```
TASK1/
├── task1_classification.ipynb    # Main analysis notebook
├── train.csv                    # Training dataset (893 passengers)
└── test.csv                     # Test dataset (420 passengers)
```

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas scikit-learn matplotlib seaborn
```

### Execute the Analysis
1. Open `task1_classification.ipynb` in Jupyter Notebook or VS Code
2. Run cells sequentially from top to bottom
3. Each cell is well-documented and explains the process
4. View visualizations and results inline

### Notebook Structure
1. **Data Loading & Exploration** - Understand the dataset
2. **Feature Visualization** - Distribution and contour plots
3. **Data Preprocessing** - Simple, beginner-friendly approach
4. **Model Training** - Compare three algorithms
5. **Parameter Tuning** - Simple manual tuning (no complex GridSearch)
6. **Final Comparison** - Clean results summary

## 📈 Key Results

### Final Model Configuration
- **Logistic Regression**: Unscaled (scaling did not help)
- **Decision Tree**: Default settings (max_depth=None was optimal)
- **Random Forest**: 50 trees (accuracy didn't change much with more trees)

### Performance Metrics
The notebook displays a clean comparison table with:
- **Accuracy**: Overall correctness percentage
- **Precision**: Of predicted survivors, how many actually survived?
- **Recall**: Of actual survivors, how many were correctly predicted?
- **ROC-AUC**: Model's ability to rank survivors vs non-survivors

## 🎓 Learning Outcomes

### Bias-Variance Tradeoff Understanding
- **High Bias, Low Variance**: Logistic Regression (simple, consistent)
- **Low Bias, High Variance**: Decision Tree (flexible, can overfit)
- **Balanced**: Random Forest (reduces variance through ensemble)

### Scaling Insights
- **When needed**: Logistic Regression with features of different ranges
- **When not needed**: Tree-based models (they split at different points)
- **Real example**: Fare (0-500) vs Pclass (1-3) range differences

### Model Selection Criteria
- **Simplicity**: Logistic Regression for interpretable results
- **Flexibility**: Decision Tree for capturing non-linear patterns
- **Robustness**: Random Forest for balanced performance
