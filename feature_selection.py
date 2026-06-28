from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pandas as pd

df = pd.read_csv("datasets/GlobalDataOnSustainableEnergy_FirstClean.csv")

df_limpo = df.drop(columns=[
    "Electricity_from_renewables_(TWh)",
    "Entity"
])

y = df["Electricity_from_renewables_(TWh)"]
       

corr_matrix = df_limpo.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

colunas_remover = [
    coluna
    for coluna in upper.columns
    if any(upper[coluna] > 0.90)
]

print(colunas_remover)

selector = SelectKBest(
    score_func=mutual_info_regression,
    k="all"
)

selector.fit(df_limpo,y)

scores = selector.scores_

resultados = pd.DataFrame({
    "Variavel": df_limpo.columns,
    "Scores": scores
})

resultados = resultados.sort_values(
    by="Scores",
    ascending=False
)

print(resultados)

selector = SelectKBest(
    score_func=mutual_info_regression,
    k=12
)

atributos_select = selector.fit_transform(df_limpo, y)

mascara = selector.get_support()

colunas = df_limpo.columns[mascara]

print(colunas)

atributos = [
    "Access_to_electricity_(%_of_population)",
    "Access_to_clean_fuels_for_cooking",
    "Renewable_energy_share_in_the_total_final_energy_consumption_(%)",
    "Electricity_from_fossil_fuels_(TWh)",
    "Electricity_from_nuclear_(TWh)",
    "Low-carbon_electricity_(%_electricity)",
    "Primary_energy_consumption_per_capita_(kWh-person)",
    "Energy_intensity_level_of_primary_energy_(MJ-$2017_PPP_GDP)",
    "Value_co2_emissions_kt_by_country",
    "gdp_growth",
    "gdp_per_capita",
    "Density-n(P-Km2)",
    "Land_Area(Km2)"
]


df_limpo[atributos].to_csv("dataset/GlobalDataOnSustainableEnergy_Selecionado.csv", index=False)