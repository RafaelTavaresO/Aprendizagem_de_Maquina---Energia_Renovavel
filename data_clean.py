import pandas as pd

df = pd.read_csv("datasets/GlobalDataOnSustainableEnergy_Raw.csv")

c = df.columns.tolist()

# Elimina variaveis que estejam nulas em mais de 25% das linhas do dataset
for coluna in df.columns:
    if df[coluna].isnull().mean() > 0.25:
        c.remove(coluna)

df_copy = df[c].copy()

# Elimina os restantes NaN das linhas do dataset
for coluna in df_copy.columns:
    if df_copy[coluna].isnull().sum() == 0:
        continue

    if df_copy[coluna].dtype == "object":
        moda = df_copy[coluna].mode()[0]
        df_copy[coluna] = df_copy[coluna].fillna(moda)
    
    else:
        mediana = df_copy[coluna].median() 
        df_copy[coluna] = df_copy[coluna].fillna(mediana)

# Altera a escrita dos nomes das colunas do dataset
df_copy.columns = (
    df_copy.columns
        .str.replace(" ", "_")
        .str.replace("/", "-", regex=False)
)

# Altera a maneira como valores acima de 1000 são armazenados para facilitar a conversão para números
for line in df_copy["Density-n(P-Km2)"]:
    if "," in line: 
        df_copy["Density-n(P-Km2)"] = df_copy["Density-n(P-Km2)"].replace(line, line.replace(",", ""))
        

# Converte a coluna de Density para números 
df_copy["Density-n(P-Km2)"] = pd.to_numeric(
     df_copy["Density-n(P-Km2)"],
     errors="coerce"
)

# Cria um CSV com o dataset limpo
df_copy.to_csv("datasets/GlobalDataOnSustainableEnergy_FirstClean.csv", index=False)
