# %%

import pandas as pd
import sqlalchemy
import mlflow

con = sqlalchemy.create_engine("sqlite:///../../data/analytics/database.db")

mlflow.set_tracking_uri("http://localhost:5000")

# %%

versions = mlflow.search_model_versions(filter_string="name='model_fiel'")
last_version = max([int(i.version) for i in versions])
model = mlflow.sklearn.load_model(f"models:///model_fiel/{last_version}")

# %%

model
# %%

data = pd.read_sql("SELECT * FROM abt_fiel", con)

predict = model.predict_proba(data[model.feature_names_in_])[:,1]

data["predict"] = predict

data
# %%
