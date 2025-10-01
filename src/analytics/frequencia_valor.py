# %%

import pandas as pd
import sqlalchemy

import matplotlib.pyplot as plt

# %%

# abre conexão com o database transacional
engine = sqlalchemy.create_engine("sqlite:///../../data/loyalty-system/database.db")

# %%

def import_query(path):
    with open(path) as open_file:
        return open_file.read()
    
query = import_query("frequencia_valor.sql")


# %%
df = pd.read_sql(query, engine)
df.head()

df = df[df['qtdePontosPos'] < 4000]

# %%

plt.plot(df['qtdeFrequencia'], df['qtdePontosPos'], 'o')
plt.grid(True)
plt.xlabel("Frequencia")
plt.ylabel("Valor")
plt.show()

# %%

from sklearn import cluster
from sklearn import preprocessing

minmax = preprocessing.MinMaxScaler()
X = minmax.fit_transform(df[['qtdeFrequencia','qtdePontosPos']])

df_X = pd.DataFrame(X, columns=['normFreq', 'normValor'])
df_X
# %%

kmean = cluster.KMeans(n_clusters=5,
                       random_state=42,
                       max_iter=1000)

kmean.fit(X)

df['cluster_calc'] = kmean.labels_
df_X['cluster'] = kmean.labels_

df.groupby(by='cluster_calc')['IdCliente'].count()

# %%

import seaborn as sns

sns.scatterplot(data=df,
                x="qtdeFrequencia",
                y="qtdePontosPos",
                hue="cluster_calc",
                palette="deep")

plt.hlines(y=1500, xmin=0,xmax=25, colors='black')
plt.hlines(y=750, xmin=0,xmax=25, colors='black')
plt.vlines(x=4, ymin=0,ymax=750, colors='black')
plt.vlines(x=10, ymin=0,ymax=3000, colors='black')
plt.grid()

# %%

sns.scatterplot(data=df,
                x="qtdeFrequencia",
                y="qtdePontosPos",
                hue="cluster",
                palette="deep")

plt.grid()
# %%
