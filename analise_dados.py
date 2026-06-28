from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from scipy.stats import kurtosis
import numpy as np
import plotly.express as px
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_raw/GlobalDataOnSustainableEnergy_FirstClean.csv")

#print(df.info())
#print(df.describe())

# for id, coluna in enumerate(df.columns):
#     if id == 14:
#         df[coluna] = (
#             df[coluna].astype(str).str.replace(",", "", regex=False)
#         )

#         df[coluna] = pd.to_numeric(
#             df[coluna],
#             errors="coerce"
#         )

#     plt.hist(df[coluna], bins=30)
#     plt.title("Distribuição de " + coluna)
#     plt.xlabel(coluna)
#     plt.ylabel("Frequência")
#     plt.savefig("graficos/Histograma_coluna" + str(id) + ".png")
#     #plt.show()
#     plt.clf()
#     plt.close('all')

# sns.barplot(
#     data=df,
#     x="Entity",
#     y="Renewable_energy_share_in_the_total_final_energy_consumption_(%)",
#     hue="Year"
# )
# plt.xlabel("Entidades")
# plt.ylabel("Energia renovável per capita")
# plt.savefig("graficos/Emtity+Year-ReEnergyPerCapita.png")
# plt.clf()
# plt.close('all')

# df_corr = df.copy()

# for col in df_corr.columns:
#     df_corr[col] = LabelEncoder().fit_transform(df_corr[col])

# corr = df_corr.corr(numeric_only=True)

# plt.figure(figsize=(20,16))

# sns.heatmap(
#     corr,
#     annot=True,
#     cmap="coolwarm"
# )

# plt.savefig("graficos/Correlacoes.png")
# plt.clf()
# plt.close('all')

# target = "Electricity_from_renewables_(TWh)"

# correlacao_target = (
#     df.corr(numeric_only=True)[target]
#       .sort_values(ascending=False)
# )

# print(correlacao_target)

# for col in df.select_dtypes(include="number").columns:
#     plt.figure(figsize=(6,4))
#     plt.boxplot(df[col])
#     plt.title(col)
#     plt.savefig("graficos/Outliers" + col + ".png")
#     plt.clf()
#     plt.close("all")

print("\nSKEW: ")
for col in df.select_dtypes(include="number").columns:
    print(col, skew(df[col]))

print("\nKURTOSIS")
for col in df.select_dtypes(include="number").columns:
    print(col, kurtosis(df[col]))

# sns.boxplot(x=df["Production"])
# plt.savefig("graficos/Outliers.png")
# plt.clf()
# plt.close('all')

# fig = px.scatter_3d(
#     df,
#     x="Day_of_Year",
#     y="Season",
#     z="Production",
#     color="Source"
# )

# fig.write_html(
#     "graficos/producao_interativa.html"
# )


# print(df.groupby(["Source", "Season"])["Production"].mean())

# ax = sns.barplot(
#     data=df,
#     x="Season",
#     y="Production",
#     hue="Source"
# )

# for container in ax.containers:
#     ax.bar_label(container, fmt="%.0f")

# print(ax.get_yticks())
# plt.xlabel("Estações")
# plt.ylabel("Produção/MWh")
# plt.savefig("graficos/Season+Source-Production.png")
# plt.clf()
# plt.close('all')

# df_corr = df.copy()

# for col in ["Season","Month_Name","Day_Name","Source"]:
#     df_corr[col] = LabelEncoder().fit_transform(df_corr[col])

# corr = df_corr.corr(numeric_only=True)

# plt.figure(figsize=(10,8))

# sns.heatmap(
#     corr,
#     annot=True,
#     cmap="coolwarm"
# )

# plt.savefig("graficos/Correlacoes.png")
# plt.clf()
# plt.close('all')

# pivot = df.pivot_table(
#     values="Production",
#     index="Season",
#     columns="Source",
#     aggfunc="mean"
# )

# sns.heatmap(
#     pivot,
#     annot=True,
#     fmt=".0f"
# )
# plt.savefig("graficos/Heatmap.png")