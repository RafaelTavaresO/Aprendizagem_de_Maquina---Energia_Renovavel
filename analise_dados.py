from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from scipy.stats import kurtosis
import numpy as np
import plotly.express as px
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/GlobalDataOnSustainableEnergy_FirstClean.csv")

print(df.info())
print(df.describe())

for coluna in df.columns:
    if id == 14:
        df[coluna] = (
            df[coluna].astype(str).str.replace(",", "", regex=False)
        )

        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="coerce"
        )

    plt.hist(df[coluna], bins=30)
    plt.title("Distribuição de " + coluna)
    plt.xlabel(coluna)
    plt.ylabel("Frequência")
    plt.savefig("graficos/Histograma" + coluna + ".png")
    #plt.show()
    plt.clf()
    plt.close('all')

sns.barplot(
    data=df,
    x="Entity",
    y="Electricity_from_renewables_(TWh)",
    hue="Year"
)
plt.xlabel("Entidades")
plt.ylabel("Eletrecidade por Energia renovável (TWh)")
plt.savefig("graficos/Emtity+Year-ElectricityFromRenawables.png")
plt.clf()
plt.close('all')

df_corr = df.copy()

for col in df_corr.columns:
    df_corr[col] = LabelEncoder().fit_transform(df_corr[col])

corr = df_corr.corr(numeric_only=True)

plt.figure(figsize=(20,16))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.savefig("graficos/Correlacoes.png")
plt.clf()
plt.close('all')

target = "Electricity_from_renewables_(TWh)"

correlacao_target = (
    df.corr(numeric_only=True)[target]
      .sort_values(ascending=False)
)

print(correlacao_target)

for col in df.select_dtypes(include="number").columns:
    plt.figure(figsize=(6,4))
    plt.boxplot(df[col])
    plt.title(col)
    plt.savefig("graficos/Outliers" + col + ".png")
    plt.clf()
    plt.close("all")

print("\nSKEW: ")
for col in df.select_dtypes(include="number").columns:
    print(col, skew(df[col]))

print("\nKURTOSIS")
for col in df.select_dtypes(include="number").columns:
    print(col, kurtosis(df[col]))
