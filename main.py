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



def train_model(X_train, X_val, y_train, y_val):

    random_model = RandomForestClassifier()

    random_model.fit(X_train, y_train)
    y_pred = random_model.predict(X_val)

    confusion_matrix(y_val, y_pred)

    accuracy_score(y_val, y_pred) * 100