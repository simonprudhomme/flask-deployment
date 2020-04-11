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

resp = requests.post("http://0.0.0.0:8000/predict", data = json.dumps(data),
                    headers= header)

print(resp.status_code)
resp.json()

