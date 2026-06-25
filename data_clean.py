from sklearn.preprocessing import LabelEncoder
import numpy as np
import plotly.express as px
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_raw/EnergyProductionDataset.csv")

#df = df.dropna()
#print(df.info)

#print(df["Production"].describe())
#df.describe()

estatisticas = pd.DataFrame({
    "media": df.mean(numeric_only=True),
    "mediana": df.median(numeric_only=True),
    "desvio": df.std(numeric_only=True),
    "min": df.min(numeric_only=True),
    "max": df.max(numeric_only=True)
})
print(df["Production"].median(numeric_only=True),)

plt.hist(df["Production"], bins=30)
plt.title("Distribuição da Produção")
plt.xlabel("Produção")
plt.ylabel("Frequência")
plt.savefig("graficos/Histograma.png")
#plt.show()
plt.clf()
plt.close('all')

sns.boxplot(x=df["Production"])
plt.savefig("graficos/Outliers.png")
plt.clf()
plt.close('all')

fig = px.scatter_3d(
    df,
    x="Day_of_Year",
    y="Season",
    z="Production",
    color="Source"
)

fig.write_html(
    "graficos/producao_interativa.html"
)


print(df.groupby(["Source", "Season"])["Production"].mean())

ax = sns.barplot(
    data=df,
    x="Season",
    y="Production",
    hue="Source"
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f")

print(ax.get_yticks())
plt.xlabel("Estações")
plt.ylabel("Produção/MWh")
plt.savefig("graficos/Season+Source-Production.png")
plt.clf()
plt.close('all')

df_corr = df.copy()

for col in ["Season","Month_Name","Day_Name","Source"]:
    df_corr[col] = LabelEncoder().fit_transform(df_corr[col])

corr = df_corr.corr(numeric_only=True)

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.savefig("graficos/Correlacoes.png")
plt.clf()
plt.close('all')

pivot = df.pivot_table(
    values="Production",
    index="Season",
    columns="Source",
    aggfunc="mean"
)

sns.heatmap(
    pivot,
    annot=True,
    fmt=".0f"
)
plt.savefig("graficos/Heatmap.png")