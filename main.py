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


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


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
    y_processed = column_or_1d(y.map({'Yes': 1, 'No': 0}).values.reshape(-1, 1), warn = True)


    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, stratify=y_processed, random_state=1121218
    )

    return X_train, X_test, y_train, y_test



def train_model(X_train, X_test, y_train, y_test):

    xgb_cl = xgb.XGBClassifier()
    log_cl = LogisticRegression()

    xgb_cl.fit(X_train, y_train)
    log_cl.fit(X_train, y_train)

    preds = xgb_cl.predict(X_test)
    log_cl_preds = log_cl.predict(X_test)

    acc_score_xgb = accuracy_score(y_test, preds)
    precision_xgb = precision_score(y_test, preds)
    recall_xgb  = recall_score(y_test, preds)
    f1_score_xgb = f1_score(y_test, preds)


    acc_score_log_cl = accuracy_score(y_test, log_cl_preds)
    precision_log_cl = precision_score(y_test, log_cl_preds)
    recall_log_cl  = recall_score(y_test, log_cl_preds)
    f1_score_log_cl = f1_score(y_test, log_cl_preds)


    metrics_results = {

        "Accuracy Score XGB" : acc_score_xgb,
        "Precision XGB" : precision_xgb,
        "Recall XGB" : recall_xgb,
        "F1 Score XGB" : f1_score_xgb,

        "Accuracy Score LOG CLF" : acc_score_log_cl,
        "Precision LOG CLF" : precision_log_cl,
        "Recall LOG CLF" : recall_log_cl,
        "F1 Score LOG CLF" : f1_score_log_cl
    }
    return metrics_results, preds


def logging_experiments_comet(metrics_results, y_test, preds):

    """
    Looking at precision and recall
    """
    
    experiment._log_metrics(metrics_results)
    experiment.log_confusion_matrix(y_true=y_test, y_predicted=preds)
    experiment.log_parameter("C", 2)


if __name__ == "__main__":
    X, y = read_dataset()
    X_train, X_test, y_train, y_test = create_pipeline(X, y)
    metrics_results, preds = train_model(X_train, X_test, y_train, y_test)
    logging_experiments_comet(metrics_results, y_test, preds)
