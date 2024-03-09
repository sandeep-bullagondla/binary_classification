# %% importing libraries
# pandas for data loading, manipulation and analysis
import pandas as pd 
import numpy as np 
# joblib & pickle for model serialization
import joblib 
import pickle
# Logistic Regression model
from sklearn.linear_model import LogisticRegression
# Decision tree model
from sklearn.tree import DecisionTreeClassifier
# Random forest model 
from sklearn.ensemble import RandomForestClassifier 
# Xgboost
import xgboost 
# for GridSearchCV for hyper parameter tuning, train_test_split for data splitting
from sklearn.model_selection import GridSearchCV, train_test_split
# for data visulaization
import seaborn as sns 
import matplotlib.pyplot as plt 
from typing import Any
# for math operations 
import math 
# for standardizing data for model training
from sklearn.preprocessing import StandardScaler 
# for column transformations
from sklearn.compose import ColumnTransformer 
# pipeline to perform transformation
from sklearn.pipeline import Pipeline
# for model evaluation using accuracy , f1-score and confusion matrix
from sklearn .metrics import accuracy_score, f1_score, confusion_matrix
# for label encoding of multicolumns from custom_transformer.py
from custom_transformer import MultiColumnLabelEncoder
# warnings
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")


#%% Loading dataset 
def load_dataset(path:str) -> Any:
    """ 
    - path: location where data is located 
    function: Loads data from location using pandas 

    returns dataset
    """
    try: 
        # loading data from path
        data = pd.read_csv(path)
        # returning data
        return data 
    # if file not found error in above block
    except FileNotFoundError as e: 
        print(f"File not found error: {str(e)}")
    # if any other unknown exception
    except Exception as e: 
        print(f"Error: {str(e)}")
        
# Path where data is located
data_path = 'resources/data/'
# calling load_dataset function to load data
data = load_dataset(data_path+'train.csv')
# first five observation of dataset
data.head() 

# %% information of dataset 
data.info()

# describe datset
data.describe()

# %% missing values 

# function for missing values
def missing_values(data: pd.DataFrame) -> Any: 
    """ 
    - data : dataset of analysis 
    function - used to find missing values in dataset 
    returns missing values in dataset
    """
    try: 
        # to get sum of null values in dataset
        values = data.isnull().sum() 
        # returning missing values 
        return values[values>0] 
    except SyntaxError as e: 
        # if any syntax error in above block
        print(f" Syntax Error: {str(e)}")
        return e 
    except Exception as e: 
        # if any other exception in above block
        print(f"Error: {str(e)}")
        return e

# calling missing_values function to find missing values in data
missing_values(data)

# %% Inferential statistics of columns 
# mean and median of column
def inferential_statistics(data, column):
    """ 
    - data : Dataset of analysis 
    - column: variable 
    returns mean and median of the variable
    """
    try: 
        # mean of the column
        mean = data[column].mean()
        # median of the column
        median = data[column].median()
        # returning mean and median
        return mean, median  
    except Exception as e: 
        # if any other exception in above block
        print(f"Error: {str(e)}")
        return e

# iterating through all variables that are int & float data types
for var in data.select_dtypes(include=['int','float']).columns:
    # calling function for mean and median of the variable
    mean, median = inferential_statistics(data, var)
    # printing mean and median of variable
    print(f"Mean and Median of {var} are: {mean}, {median} ")

# %% density plot for numerical data
# for flattening axis
def trim_axs(axs:pd.array, N:int)-> Any:
    # flatenning axes
    axs = axs.flat
    # iterating after N number
    for ax in axs[N:]:
        # removing axes
        ax.remove()
    return axs[:N]

def dist_plots(data: pd.DataFrame, num_vars: pd.array) -> None: 

    """  
    data - dataset of analysis 
    num_vars - Numerical variables 
    returns - none
    function - generates Density plot of numerical variables
    """
    try:
        #columns of plot 
        ncols = 3
        # rows of plot based on variables and number of columns
        nrows = int(math.ceil(len(num_vars)/ncols))
        # making subplots
        f,axs = plt.subplots(nrows, ncols, figsize = (15, 10), constrained_layout = True)
        # calling trim-axs function
        axs = trim_axs(axs, len(num_vars))
        # rotating x labels by 45 degrees
        plt.xticks(rotation = 45)
        # title of the main plot
        f.suptitle('Density Plots for Numerical variables')
        # iterating through axs and variables
        for ax, var in zip(axs, num_vars):
            # density plot for the variable
            sns.distplot(x=data[var], ax=ax)
            # title for subplot
            ax.set_title(var)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    except Exception as e: 
        # if any other exception in above block
        print(f"Error: {str(e)}")
        return e

num_vars = data.select_dtypes(include=['int','float']).columns
dist_plots(data, num_vars)

# %%  count plots 
def count_plots(data: pd.DataFrame, cat_vars: pd.array) -> None: 

    """  
    data - dataset of analysis 
    cat_vars - Categorical variables 
    returns - none
    function - generates count plot of categorical variables
    """
    try:
        #columns of plot 
        ncols = 3
        # rows of plot based on variables and number of columns
        nrows = int(math.ceil(len(cat_vars)/ncols))
        # making subplots
        f,axs = plt.subplots(nrows, ncols, figsize = (15, 5), constrained_layout = True)
        # calling trim-axs function
        axs = trim_axs(axs, len(cat_vars))
        # rotating x labels by 45 degrees
        plt.xticks(rotation = 45)
        # title of the main plot
        f.suptitle('Count Plots for categorical variables')
        # iterating through axs and variables
        for ax, var in zip(axs, cat_vars):
            # count plot for the variable
            sns.countplot(x=data[var], ax=ax)
            # title for subplot
            ax.set_title(var)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    except Exception as e: 
        # if any other exception in above block
        print(f"Error: {str(e)}")
        return e

# categorical variables in dataset
cat_vars = data.select_dtypes(include="object").columns 
# countplot for categorical dataset
count_plots(data, cat_vars) 

# %% imbalance in data 
def imbalance_check(data:pd.DataFrame, target: str) -> Any:
    """  
    data - data set of analysis 
    target - target variable 
    function - Gives value counts of target variable in data 
    return value counts
    """
    try: 
        # value counts of target variable
        counts = data[target].value_counts() 
        return counts
    except Exception as e: 
        # if any error in above block
        print(f"Error: {str(e)}")

# calling imbalance_check if there is imbalance in data
imbalance_check(data, 'class') 

#%% data splitting 

 
def split_data(data: pd.DataFrame, drop_columns:pd.array,target:str) -> Any: 
    """  
    data - dataset of analysis 
    drop_columns - columns to be removed from training data 
    target - target variable of model 
    function - Splits data into 80% training data and 20% testing data 
    returns Splitted train and test datasets
    """
    try: 
        # removing target varaible from data for X
        X = data.drop(drop_columns, axis= 1) 
        # assining only target variable data to y
        y = data[target] 
        # splitting data into train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42) 
        return X_train, X_test, y_train, y_test 
    except Exception as e: 
        # if any error in above block
        print(f"Error: {str(e)}") 

# columns to be dropped from training data
drop_columns = ['X_6', 'X_17', 'class']
# target variable
target = 'class'
# splitting data by calling split_data function  
X_train, X_test, y_train, y_test = split_data(data, drop_columns, target) 

# %% Data transformation 
def data_transformers(data:pd.DataFrame) -> Pipeline:
    """  
    data - dataset of analysis 
    function :  
        transforms numerical and categorical data using StandardScaler and LabelEncoder with the help of Pieline
    """
    try: 
        # List of columns to be scaled and encoded
        numeric_columns = data.select_dtypes(include=['int','float']).columns
        categorical_columns = data.select_dtypes(include='object').columns
        # Define the preprocessing steps using ColumnTransformer and Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', StandardScaler(), numeric_columns),
                ('categorical', Pipeline([
                    ( 'encoder', MultiColumnLabelEncoder(columns=categorical_columns))
                ]), categorical_columns)
            ])
        # fitting preprocessor in pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
        ])
        return pipeline  
    except Exception as e: 
        # if any error in above block
        print(f"Error: {str(e)}")

# calling data_transformers function
pipeline = data_transformers(X_train)
# Fit and transform on the training data
X_train = pipeline.fit_transform(X_train)
# Transform on the testing data (Note: only transform, not fit)
X_test = pipeline.transform(X_test)
# dumping pipeline for preprocessing
joblib.dump(pipeline, 'resources/models/pipeline.joblib') 

# %% 
# Prediction function
def prediction(model: Any, X:Any, y:Any) -> Any: 
    """ 
    model: Model to which predictions to be performed 
    X: test data with independent variables 
    y: test data with target variable 
    function: 
        Predicts labes for X, and also performs accuracy and F1-score for the predicted and actual labels
    returns predicted values, accuracy and f1-score
    """
    try: 
        # predicting values for model 
        predictions = model.predict(X)
        # accuracy score for model
        accuracy = accuracy_score(predictions, y)
        print(f" For {model}")
        print(f'Accuracy: {accuracy:.4f}') 
        # F1-score for model
        f1score = f1_score(predictions, y)
        print(f'F1-score: {f1score:.4f}') 
        return predictions, accuracy, f1score
    except Exception as e: 
        # if any error in above block
        print(f"Error: {str(e)}")

# Confusion matrix 
    
def confusion_mat(actual: Any, predictions:Any) -> None: 
    """  
    Actual - Actual values of data 
    predictions - predicted values of data 
    function - Generates Confusion matrix plot for the model
    """
    try: 
        # Confusion Matrix
        conf_matrix = confusion_matrix(actual, predictions)
        # heat map for confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        # plot title
        plt.title('Confusion Matrix')
        # x and y labels
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # show plot
        plt.show()
    except Exception as e: 
        # if any error in above block
        print(f"Error: {str(e)}")
# %% Extend the pipeline with Logistic Regression
def logistic_regression(X: Any, y:Any) -> Any:
    """ 
    X - Training data with independent variables 
    y - training data witn target variables 

    function - Fits Logistic Regression Model for data
    """
    try: 
        # define Logistic regression model
        model = LogisticRegression()
        # fitting logistic regression model
        model.fit(X, y)
        return model
    except Exception as e: 
        # if any error in above block
        print(f"Error: {str(e)}")

# calling logistic_regression function to fit Logistic regression model
lr = logistic_regression(X_train, y_train)
# Predictions, accuracy and F1-score on the testing data for Logistic resgression model
y_pred_lg, accuracy_lg, f1Score_lg = prediction(lr, X_test, y_test)
# confusion matrix for logistic regression model
confusion_mat(y_test, y_pred_lg)

joblib.dump(lr, 'resources/models/logistic_regression.joblib')
# %%  Grid Search for Hyperparameter tuning of models 
# function for GridSearch 
def grid_search(model:Any, param_grid:Any, X:Any, y:Any) -> Any: 
    """ 
    model - Model to which hyperparameter tuning is to be performed 
    param_grid: parameters of the model
    X - Training data with independent variables 
    y - training data witn target variables 

    function - Fits GridSearchCv and get best hyperparameters for the model 
    returns best estimator and hyperparameter of model
    """
    try: 
        # defining GridSearchCV for model with parameters and cross-validation 5
        grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        # Fit the grid search on the data
        grid.fit(X, y) 
        # retuning best estimator and parameters
        return  grid.best_estimator_, grid.best_params_
    except Exception as e: 
        # if any error in above block
        print(f"Error: {str(e)}")

# %% Decision Tree Classifier 

def decision_tree(X:Any, y:Any) -> Any:
    """ 
    X - Training data with independent variables 
    y - training data witn target variables 

    function - Fits Decision Tree classifier for data
    returns - best model and best hyperparametrs for decision tree model
    """
    try: 
        # Define a Decision Tree classifier
        dt_classifier = DecisionTreeClassifier()
        # Define the hyperparameter grid
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        # best estimator and best hyperparamters for decision tree using grid_search
        best_estimator, best_parameters =   grid_search(dt_classifier, param_grid, X, y)
        return best_estimator, best_parameters 
    except Exception as e: 
        # if any error in above block
        print(f"Error: {str(e)}")

# Get the best model & parameters for Decision Tree
dt_model, dt_params = decision_tree(X_train, y_train)
# Display the best hyperparameters for decision tree
print('Decision Tree Best Hyperparameters:', dt_params)
# Predictions, accuracy and F1-score on the testing data for Decision Tree model
y_pred_dt, accuracy_dt, f1Score_dt = prediction(dt_model, X_test, y_test)
# confusion matrix for Decision tree model
confusion_mat(y_test, y_pred_dt)
# dumping decision tree model 
joblib.dump(dt_model, 'resources/models/dt_model.joblib')

#%% Random Forest Classifier
def random_forest(X:Any, y:Any) -> Any:
    """ 
    X - Training data with independent variables 
    y - training data witn target variables 

    function - Fits Decision Tree classifier for data
    returns - best model and best hyperparametrs for Random Forest model
    """
    try: 
        # Defining Random Forest classifier
        rf = RandomForestClassifier(random_state=42)
        # Define the hyperparameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        # best estimator and best hyperparamters for random forest using grid_search
        best_estimator, best_parameters =   grid_search(rf, param_grid, X, y)
        return best_estimator, best_parameters
    except Exception as e: 
        # if any error in above block
        print(f"Error: {str(e)}")

# Get the best model & parameters for random Forest
random_forest_model, random_forest_params = random_forest(X_train, y_train)
# Display the best hyperparameters for rndom Forest
print('Best Hyperparameters for Random Forest:', random_forest_params)
# Predictions, accuracy and F1-score on the testing data for random forest  model
y_pred_rf, accuracy_rf, f1Score_rf = prediction(random_forest_model, X_test, y_test)
# confusion matrix for Random forest model
confusion_mat(y_test, y_pred_rf)
# dumping random forest model 
joblib.dump(random_forest_model, 'resources/models/random_forest_model.joblib')

# %% XGBoost Classifier 

def xgboost_classifier(X:Any, y:Any)-> Any: 
    """ 
    X - Training data with independent variables 
    y - training data witn target variables 

    function - Fits Decision Tree classifier for data
    returns - best model and best hyperparametrs for XGBoost model
    """
    try: 
        #defining xgboost classifier
        xgb_classifier = xgboost.XGBClassifier(scale_pos_weight=1.001)
        # Define the hyperparameter grid for XGBclassifier
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 1]
        }
        # best estimator and best hyperparamters for XGBClassifier using grid_search
        best_estimator, best_parameters =   grid_search(xgb_classifier, param_grid, X, y)
        return best_estimator, best_parameters
    except Exception as e: 
        # if any error in above block
        print(f"Error: {str(e)}")

# Get the best model & parameters for XGBoost
xgboost_model, xgboost_params = xgboost_classifier(X_train, y_train)
# Display the best hyperparameters
print('Best Hyperparameters for XGBoost:', xgboost_params)
# Predictions, accuracy and F1-score on the testing data for XGBoost  model
y_pred_xg, accuracy_xg, f1Score_xg = prediction(xgboost_model, X_test, y_test)
# confusion matrix for XGBoost model
confusion_mat(y_test, y_pred_xg)
# dumping random forest model 
joblib.dump(xgboost_model, 'resources/models/xgboost_model.joblib')

# %% Validation dataset 
# Loading Validation dataset
val_data = load_dataset(data_path + "validation.csv")
# dropping columns from validation data 
X_val = val_data.drop(drop_columns, axis=1)
# assing target variable data
y_val = val_data[target] 

# transforming validation data using pipeline
X_val = pipeline.transform(X_val) 

# Having all models in array
models = [lr, dt_model, random_forest_model, xgboost_model]
# itearating through each model
for model in models: 
    # prediction, accuracy and F1-Score for each model for Validation data
    y_val_preds, accuracy_val, f1score_val = prediction(model, X_val, y_val)
    #Confusion matrix for each model
    confusion_mat(y_val, y_val_preds)

#%% 
# Library for auc and roc_curve
from sklearn.metrics import roc_curve, auc  
def au_roc_plot(model: Any) -> Any: 
    """ 
    model - Machine learning model
    function - generates Roc plot for model

    """
    try:
        # prediction probabilities of model
        y_pred_proba = model.predict_proba(X_val)[:,1]
        # false positive and true positive rates
        fpr, tpr, _ = roc_curve(y_val.iloc[:], y_pred_proba)
        # Area under curve for true and false positive rates
        roc_auc = auc(fpr, tpr)

        plt.figure()
        # plotting fpr and tpr
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for model')
        plt.legend(loc='lower right')
        plt.show()
    except Exception as e: 
        print(f"Error: {str(e)}")

# Iterating through all models
for model in models: 
    # AUROC plot for model
    au_roc_plot(model)