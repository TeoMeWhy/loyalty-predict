# %%

import pandas as pd
import sqlalchemy

from sklearn import model_selection

from feature_engine import selection
from feature_engine import imputation
from feature_engine import encoding

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

con = sqlalchemy.create_engine("sqlite:///../../data/analytics/database.db")

# %%

## SAMPLE - IMPORT DOS DADOS

df = pd.read_sql("select * from abt_fiel", con)
df.head()

# %%

# SAMPLE - OOT

df_oot = df[df['dtRef']==df['dtRef'].max()].reset_index(drop=True)
df_oot

# %%

# SAMPLE - Teste e Treino

target = 'flFiel'
features = df.columns.tolist()[3:]

df_train_test = df[df['dtRef']<df['dtRef'].max()].reset_index(drop=True)

y = df_train_test[target]   # Isso é um pd.Series (vetor)
X = df_train_test[features] # Isso é um pd.DataFrame (matriz)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Base Treino: {y_train.shape[0]} Unid. | Tx. Target {100*y_train.mean():.2f}%")
print(f"Base Test: {y_test.shape[0]} Unid. | Tx. Target {100*y_test.mean():.2f}%")

# %%

# EXPLORE - MISSING

s_nas = X_train.isna().mean()
s_nas = s_nas[s_nas>0]

s_nas

# %%

## EXPLORE BIVARIADA

cat_features = ['descLifeCycleAtual', 'descLifeCycleD28']
num_features = list(set(features) - set(cat_features))

df_train = X_train.copy()
df_train[target] = y_train.copy()

df_train[num_features] = df_train[num_features].astype(float)

bivariada = df_train.groupby(target)[num_features].median().T
bivariada['ratio'] = (bivariada[1] + 0.001) / (bivariada[0]+0.001)
bivariada = bivariada.sort_values(by='ratio', ascending=False)
bivariada

# %%
df_train.groupby('descLifeCycleAtual')[target].mean()

# %%
df_train.groupby('descLifeCycleD28')[target].mean()

# %%

# MODIFY - DROP

X_train[num_features] = X_train[num_features].astype(float)

to_remove = bivariada[bivariada['ratio']==1].index.tolist()
drop_features = selection.DropFeatures(to_remove)

# MODIFY - MISSING

fill_0 = ['github2025', 'python2025']
imput_0 = imputation.ArbitraryNumberImputer(arbitrary_number=0,
                                            variables=fill_0)

imput_new = imputation.CategoricalImputer(
    fill_value='Nao-Usuario',
    variables=['descLifeCycleD28'])

imput_1000 = imputation.ArbitraryNumberImputer(
    arbitrary_number=1000,
    variables=['avgIntervaloDiasVida',
               'avgIntervaloDiasD28',
               'qtdDiasUltiAtividade'],
    )

# MODIFY - ONEHOT

onehot = encoding.OneHotEncoder(variables=cat_features)

# MODIFY - APLICANDO TRANSFORMAÇÕES NO DATASET
 
X_train_transform = drop_features.fit_transform(X_train)
X_train_transform = imput_0.fit_transform(X_train_transform)
X_train_transform = imput_new.fit_transform(X_train_transform)
X_train_transform = imput_1000.fit_transform(X_train_transform)
X_train_transform = onehot.fit_transform(X_train_transform)

# %%

X_train_transform.head()

# %%

# MODEL

from sklearn import tree
from sklearn import ensemble

# model = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=50)
model = ensemble.AdaBoostClassifier(random_state=42,
                                    n_estimators=150,
                                    learning_rate=0.1)

model.fit(X_train_transform, y_train)

# %%

# ASSESS

from sklearn import metrics

y_pred_train = model.predict(X_train_transform)
y_proba_train = model.predict_proba(X_train_transform)

acc_train = metrics.accuracy_score(y_train, y_pred_train)
auc_train = metrics.roc_auc_score(y_train, y_proba_train[:,1])

print("Acurácia Treino:", acc_train)
print("AUC Treino:", auc_train)

# %%

X_test_transform = drop_features.transform(X_test)
X_test_transform = imput_0.transform(X_test_transform)
X_test_transform = imput_new.transform(X_test_transform)
X_test_transform = imput_1000.transform(X_test_transform)
X_test_transform = onehot.transform(X_test_transform)

y_pred_test = model.predict(X_test_transform)
y_proba_test = model.predict_proba(X_test_transform)

acc_test = metrics.accuracy_score(y_test, y_pred_test)
auc_test = metrics.roc_auc_score(y_test, y_proba_test[:,1])

print("Acurácia Teste:", acc_test)
print("AUC Teste:", auc_test)

# %%
X_oot = df_oot[features]
y_oot = df_oot[target]

X_oot_transform = drop_features.transform(X_oot)
X_oot_transform = imput_0.transform(X_oot_transform)
X_oot_transform = imput_new.transform(X_oot_transform)
X_oot_transform = imput_1000.transform(X_oot_transform)
X_oot_transform = onehot.transform(X_oot_transform)

y_pred_oot = model.predict(X_oot_transform)
y_proba_oot = model.predict_proba(X_oot_transform)

acc_oot = metrics.accuracy_score(y_oot, y_pred_oot)
auc_oot = metrics.roc_auc_score(y_oot, y_proba_oot[:,1])

print("Acurácia OOT:", acc_oot)
print("AUC OOT:", auc_oot)


# %%

y_predict_fodase = pd.Series([0]*y_test.shape[0])
y_proba_fodase = pd.Series([y_train.mean()]*y_test.shape[0])

acc_fodase = metrics.accuracy_score(y_test, y_predict_fodase)
auc_fodase = metrics.roc_auc_score(y_test, y_proba_fodase)
print("Acurácia Fodase:", acc_fodase)
print("AUC Fodase:", auc_fodase)

# %%

features_names = X_train_transform.columns.tolist()

feature_importance = pd.Series(model.feature_importances_,
                               index=features_names)

feature_importance.sort_values(ascending=False)