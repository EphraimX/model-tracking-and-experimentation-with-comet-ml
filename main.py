import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



def preprocessing(datapath='data/corrected_fame_dataset.csv'):

    df = pd.read_csv(datapath)
    df.drop(columns=['name','released'], inplace=True)

    X =  df.drop(columns='is_movie_successful') 
    y = df['is_movie_successful']

    categorical_columns = ['rating', 'genre', 'year','director','writer','star','country','company','month_released']
    X = pd.get_dummies(data=X, columns=categorical_columns)
    y = y.map({'Yes': 1, 'No': 0})

    X_train, X_val, y_train, y_val =  train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_val, y_train, y_val

    from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


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


import xgboost as xgb

xgb_cl = xgb.XGBClassifier()

print(type(xgb_cl))


X_processed = full_processor.fit_transform(X)
y_processed = y.map({'Yes': 1, 'No': 0}).values.reshape(-1, 1)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed, stratify=y_processed, random_state=1121218
)


from sklearn.metrics import accuracy_score

xgb_cl.fit(X_train, y_train)

preds = xgb_cl.predict(X_test)

accuracy_score(y_test, preds)



def train_model(X_train, X_val, y_train, y_val):

    random_model = RandomForestClassifier()

    random_model.fit(X_train, y_train)
    y_pred = random_model.predict(X_val)

    confusion_matrix(y_val, y_pred)

    accuracy_score(y_val, y_pred) * 100