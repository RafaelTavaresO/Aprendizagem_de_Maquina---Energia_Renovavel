from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("datasets/GlobalDataOnSustainableEnergy_Selecionado.csv")

y = df["Electricity_from_renewables_(TWh)"]

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df)