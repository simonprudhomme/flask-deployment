import json
import requests
import pandas as pd

header = {'Content-Type': 'application/json', 'Accept': 'application/json'}

"""
Reading test batch
"""
df = pd.read_csv('data.csv', encoding="utf-8-sig")

"""Converting Pandas Dataframe to json
"""
data = df.to_json(orient='records')
data

resp = requests.get("http://0.0.0.0:5000/predict", data = json.dumps(data),
                    headers= header)

print(resp.status_code)
resp.json()


import requests

params = {
    'class': 2,
    'age': 22,
    'sibsp': 2,
    'parch': 0,
    'title': 'mr',
    'sex': 'male',
}

url = 'http://0.0.0.0:5000/predict'
r = requests.get(url, params)
print(r.url)
print(r.json())

