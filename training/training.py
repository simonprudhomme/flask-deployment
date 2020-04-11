# import
import pandas as pd 
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

import dill as pickle

# load data
data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')
data.head(3)

# split data
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create Pipeline
ohe = OneHotEncoder(handle_unknown='ignore')
impute = SimpleImputer(add_indicator=True)
clf =  HistGradientBoostingClassifier(max_depth= 10, max_iter=200, random_state=42)
ct = make_column_transformer(
    (ohe,['Pclass','Sex']),
    (impute, ['Age','Siblings/Spouses Aboard','Parents/Children Aboard','Fare']),
    remainder='drop'
    )
pipe = make_pipeline(ct, clf)


#Train and Evaluate Model
pipe.fit(X_train, y_train)
pipe.score(X_test,y_test)

# Save Model

filename = 'model_v1.pk'
with open('models/'+filename, 'wb') as file:
    pickle.dump(pipe, file)

with open('models/'+filename, 'rb') as f:
    loaded_model = pickle.load(f)

loaded_model.predict(X[:10])