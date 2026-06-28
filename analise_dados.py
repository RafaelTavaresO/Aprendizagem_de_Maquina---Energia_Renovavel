from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/GlobalDataOnSustainableEnergy_FirstClean.csv")

# Descreve elementos fundamentais do dataset
print(df.info())
print(df.describe())

# Gera historiogramas das variáveis do dataset
for coluna in df.columns:
    plt.hist(df[coluna], bins=30)
    plt.title("Distribuição de " + coluna)
    plt.xlabel(coluna)
    plt.ylabel("Frequência")
    plt.savefig("graficos/Histograma" + coluna + ".png")
    #plt.show()
    plt.clf()
    plt.close('all')

# Gera um Barplot entre Entidade,
sns.barplot(
    data=df,
    x="Land_Area(Km2)",
    y="Renewable_energy_share_in_the_total_final_energy_consumption_(%)",
    hue="Year"
)
plt.xlabel("Entidades")
plt.ylabel("Participação das energias renováveis (%)")
plt.savefig("graficos/Emtity+Year-Renewable_energy_share.png")
plt.clf()
plt.close('all')

# Gera um Heatmap da correlação entre as coluna do dataset
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


# Mostra um ranking de correlação entre a variável alvo e o resto das colunas 
target = "Renewable_energy_share_in_the_total_final_energy_consumption_(%)"

correlacao_target = (
    df.corr(numeric_only=True)[target]
      .sort_values(ascending=False)
)

print(correlacao_target)

# Gera graficos boxplot das colunas do dataset
for col in df.select_dtypes(include="number").columns:
    plt.figure(figsize=(6,4))
    plt.boxplot(df[col])
    plt.title(col)
    plt.savefig("graficos/Outliers" + col + ".png")
    plt.clf()
    plt.close("all")