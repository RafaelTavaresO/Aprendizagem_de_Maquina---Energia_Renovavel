import pandas as pd

df = pd.read_csv("dataset_raw/GlobalDataOnSustainableEnergy.csv")

#print(df.info())

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

df_copy.columns = (
    df_copy.columns
        .str.replace(" ", "_")
        .str.replace("/", "-", regex=False)
)

for col in df_copy.select_dtypes(include="number").columns:

    Q1 = df_copy[col].quantile(0.25)
    Q3 = df_copy[col].quantile(0.75)

    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5*IQR
    limite_superior = Q3 + 1.5*IQR

    quantidade = (
        (df_copy[col] < limite_inferior) |
        (df_copy[col] > limite_superior)
    ).sum()

    df_copy[col] = df_copy[col].clip(
        lower=limite_inferior,
        upper=limite_superior
    )

    print("Limitante inferior: " + str(limite_inferior) + " | Limitante Superior: " + str(limite_superior) + " -> " + col, str(quantidade))


df_copy.to_csv("dataset_raw/GlobalDataOnSustainableEnergy_FirstClean.csv", index=False)
