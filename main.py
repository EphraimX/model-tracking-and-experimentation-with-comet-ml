import pandas as pd
from sklearn.utils import column_or_1d
import xgboost as xgb
import comet_ml
from decouple import config


API_KEY = config('COMET_API_KEY')

experiment = comet_ml.Experiment(
    api_key=API_KEY,
    project_name="tracking-machine-learning-models-using-comet-ml"
)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



def read_dataset(datapath='data/corrected_fame_dataset.csv'):


    # read dataset
    df = pd.read_csv(datapath)
    df.drop(columns=['name','released'], inplace=True)

    # separate into target and feature 
    X =  df.drop(columns='is_movie_successful') 
    y = df['is_movie_successful']

    return X, y


def create_pipeline(X, y):


    categorical_pipeline = Pipeline(
        steps=[
            ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler())
        ]
    )

    cat_cols = X.select_dtypes(exclude="number").columns
    num_cols = X.select_dtypes(include="number").columns


    full_processor = ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, num_cols),
            ("categorical", categorical_pipeline, cat_cols)
        ]
    )


    X_processed = full_processor.fit_transform(X)
    y_processed = y.map({'Yes': 1, 'No': 0}).values.ravel()


    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, stratify=y_processed, random_state=1121218
    )

    return X_train, X_test, y_train, y_test



def train_model(models, X_train, X_test, y_train, y_test):

    metrics_results = {}

    for count in range(len(models)):

        model_name = list(models.keys())[count]
        model = list(models.values())[count]

        model.fit(X_train, y_train)
        model_prediction = model.predict(X_test)

        metrics_results[f'Accuracy Score {model_name}'] = accuracy_score(y_test, model_prediction)
        metrics_results[f'Precision Score {model_name}'] = precision_score(y_test, model_prediction)
        metrics_results[f'Recall Score {model_name}'] = recall_score(y_test, model_prediction)
        metrics_results[f'F1 Score {model_name}'] = f1_score(y_test, model_prediction)

    return metrics_results


def logging_experiments_comet(metrics_results, y_test):

    """
    Looking at precision and recall
    """
    
    experiment._log_metrics(metrics_results)
    experiment.log_parameter("C", 2)


if __name__ == "__main__":
    X, y = read_dataset()
    X_train, X_test, y_train, y_test = create_pipeline(X, y)

    models = {
        "Logistic Regression" : LogisticRegression(),
        "Decision Tree" : DecisionTreeClassifier(),
        "Random Forest" : RandomForestClassifier(),
        "Ada Boost" : AdaBoostClassifier(),
        "XGBoost" : xgb.XGBClassifier()
    }

    metrics_results = train_model(models, X_train, X_test, y_train, y_test)
    logging_experiments_comet(metrics_results, y_test)
