from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import time
import json
import pandas as pd

# Carregamento do conjunto de dados selecionado após a etapa de feature selection
df = pd.read_csv("datasets/GlobalDataOnSustainableEnergy_Selecionado.csv")

# Separação dos atributos da variável alvo
df_limpo = df.drop(columns=[
    "Renewable_energy_share_in_the_total_final_energy_consumption_(%)"
])

# Seleção dos casos de uso que serão removidos do treinamento
# Caso 1: Países com grande extensão territorial e participação moderada/alta de eletricidade proveniente de fontes de baixo carbono
escolha1 = df_limpo[
    (df_limpo["Land_Area(Km2)"] > df_limpo["Land_Area(Km2)"].quantile(0.85)) &
    (df_limpo["Low-carbon_electricity_(%_electricity)"] > df_limpo["Low-carbon_electricity_(%_electricity)"].quantile(0.40))]

# Caso 2: Países com baixa densidade populacional, emissões de CO₂ moderadas/altas e consumo de energia per capita acima da mediana
escolha2 = df_limpo[
    (df_limpo["Density-n(P-Km2)"] < df_limpo["Density-n(P-Km2)"].quantile(0.20)) &
    (df_limpo["Value_co2_emissions_kt_by_country"] > df_limpo["Value_co2_emissions_kt_by_country"].quantile(0.35)) &
    (df_limpo["Primary_energy_consumption_per_capita_(kWh-person)"] > df_limpo["Primary_energy_consumption_per_capita_(kWh-person)"].quantile(0.50))]

# Seleção aleatória e reprodutível de um exemplo de cada perfil
caso1 = escolha1.sample(1)
caso2 = escolha2.sample(1)

# Armazenamento dos valores reais da variável alvo para posterior comparação
y1 = df["Renewable_energy_share_in_the_total_final_energy_consumption_(%)"].iloc[caso1.index[0]]
y2 = df["Renewable_energy_share_in_the_total_final_energy_consumption_(%)"].iloc[caso2.index[0]]

# Remoção dos casos de uso do conjunto de treinamento
df_limpo = df_limpo.drop(caso1.index)
df_limpo = df_limpo.drop(caso2.index)

# Construção da variável alvo sem os casos utilizados na demonstração
y = df["Renewable_energy_share_in_the_total_final_energy_consumption_(%)"]
y = y.drop(caso1.index.union(caso2.index))

# Padronização dos atributos para modelos sensíveis à escala dos dados
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_limpo),
    columns=df_limpo.columns,
    index=df_limpo.index
)

# Definição do espaço de busca de hiperparâmetros da Árvore de Decisão
param_tree = {

    "max_depth":[3,5,7,8,9,10,15,20],

    "min_samples_split":[2,5,10],

    "min_samples_leaf":[1,2,4,8]
}  

# Definição do espaço de busca de hiperparâmetros do kNN
param_knn = {

    "n_neighbors": [3,5,7,9,11,13,15,17,21],

    "weights": ["uniform","distance"],

    "metric": ["euclidean","manhattan","minkowski"]
}    

# Definição do espaço de busca de hiperparâmetros da Rede Neural MLP
param_mlp = {

    "hidden_layer_sizes":[

        (50,),
        (100,),
        (100,50)

    ],

    "activation":[

        "relu",
        "tanh"

    ],

    "learning_rate":[

        "constant",
        "adaptive"

    ],

    "max_iter":[7000]
}

# Função responsável pelo treinamento e avaliação dos modelos
def treinar_modelo(modelo, parametros, atributos, y, nome_modelo):
    # """
    # Realiza a otimização de hiperparâmetros utilizando Grid Search,
    # avalia o modelo por validação cruzada e retorna as métricas
    # obtidas juntamente com o modelo treinado.
    # """
        
    # Início da medição do tempo de execução do Grid Search
    inicio_grid = time.perf_counter()
   
    # Configuração da validação cruzada k-Fold
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42 
    )

    # Configuração da busca pelos melhores hiperparâmetros
    grid = GridSearchCV(
        estimator=modelo,
        param_grid=parametros,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    # Treinamento do Grid Search
    grid.fit(atributos, y)

    # Cálculo do tempo gasto pelo Grid Search
    fim_grid = time.perf_counter()
    tempo_execucao_grid = fim_grid - inicio_grid

    # Cálculo do tempo gasto pelo Grid Search
    print(f"\n===== {nome_modelo} =====")

    print(grid.best_params_)

    # Construção de um novo modelo utilizando apenas os melhores parâmetros
    melhores_parametros = grid.best_params_
    novo_modelo = modelo.__class__(**melhores_parametros)

    # Início da medição do tempo da validação cruzada
    inicio_cross_validation = time.perf_counter()
    
    # Avaliação do modelo utilizando validação cruzada
    scores = cross_validate(
        novo_modelo,
        atributos,
        y,
        cv=kf,
        scoring={
            "RMSE":"neg_root_mean_squared_error",
            "MAE":"neg_mean_absolute_error",
            "R2":"r2"
        }
    )
    # Cálculo do tempo de execução da validação cruzada
    fim_cross_validation = time.perf_counter()
    tempo_execucao_cross_validation = fim_cross_validation - inicio_cross_validation

    # Cálculo das médias das métricas obtidas nos folds
    rmse = (-scores["test_RMSE"]).mean()
    mae = (-scores["test_MAE"]).mean()
    r2 = scores["test_R2"].mean()

    # Cálculo do desvio padrão das métricas
    rmse_std = (-scores["test_RMSE"]).std()
    mae_std = (-scores["test_MAE"]).std()
    r2_std = scores["test_R2"].std()

    # Exibição dos resultados obtidos
    print(f"RMSE : {rmse:.2f}±{rmse_std:.2f}")
    print(f"MAE  : {mae:.2f}±{mae_std:.2f}")
    print(f"R²   : {r2:.2f}±{r2_std:.2f}")

    print(f"Tempo de execução GRID: {tempo_execucao_grid:.2f} segundos")
    print(f"Tempo de execução Cross Validation: {tempo_execucao_cross_validation:.2f} segundos")
    print(f"Tempo Total (s): {(tempo_execucao_cross_validation + tempo_execucao_grid):.2f}")

    # Treinamento final do modelo utilizando todos os dados disponíveis
    novo_modelo.fit(atributos,y)

    # Organização dos resultados para exportação
    resultado = {
        "Modelo": nome_modelo,
        "RMSE": f"{rmse:.2f}±{rmse_std:.2f}",
        "MAE": f"{mae:.2f}±{mae_std:.2f}",
        "R2": f"{r2:.2f}±{r2_std:.2f}",
        "Tempo GRID (s)": f"{tempo_execucao_grid:.2f}",
        "Tempo Cross Validation (s)": f"{tempo_execucao_cross_validation:.2f}",
        "Tempo Total (s)": f"{(tempo_execucao_cross_validation + tempo_execucao_grid):.2f}",
        "Parametros": melhores_parametros
    }

    return resultado, novo_modelo

# Instanciação e treinamento da Árvore de Decisão
tree = DecisionTreeRegressor(
    #random_state=42
)

resultado_tree, modelo_arvore = treinar_modelo(
    tree,
    param_tree,
    df_limpo,
    y,
    "Árvore de Decisão"
)

# Instanciação e treinamento do kNN
knn = KNeighborsRegressor()

resultado_knn, modelo_knn = treinar_modelo(
    knn,
    param_knn,
    df_scaled,
    y,
    "kNN"
)

# Instanciação e treinamento da Rede Neural MLP
mlp = MLPRegressor(
    #random_state=42
)

resultado_mlp, modelo_mlp= treinar_modelo(
    mlp,
    param_mlp,
    df_scaled,
    y,
    "MLP"
)

# Função responsável por demonstrar um caso de uso para um modelo treinado
def demonstrar_caso(modelo, entrada, nome_modelo, caso, valor_real=None):
    # """
    # Executa a predição de um caso de uso, calcula os erros em relação
    # ao valor real e retorna os resultados para apresentação.
    # """

    # Realização da predição
    pred = modelo.predict(entrada)[0]

    print("-"*50)
    print(f"{nome_modelo} --- Caso {caso}")
    print(f"Predição: {pred:.2f}")
    
    # Cálculo do erro absoluto e percentual
    erro = abs(pred-valor_real)
    erro_percentual = erro/valor_real *100

    # Exibição dos resultados da demonstração
    print(f"Valor real: {valor_real:.2f}")
    print(f"Erro: {erro:.2f}")
    print(f"Erro percentual (%): {erro_percentual:.2f}%")

    # Organização dos resultados para exportação
    return {
        "Modelo": nome_modelo,
        "Predicao":f"{pred:.2f}",
        "Valor_Real":f"{valor_real:.2f}",
        "Erro":f"{erro:.2f}",
        "Erro Percentual (%)": f"{erro_percentual:.2f}%"
    }

# Demonstração dos casos utilizando a Árvore de Decisão
resultado_arvore_caso1 = demonstrar_caso(modelo_arvore, caso1, "Arvore de decisao", 1, y1)
resultado_arvore_caso2 = demonstrar_caso(modelo_arvore, caso2, "Arvore de decisao", 2, y2)

caso1_original = caso1.copy()
caso2_original = caso2.copy()

# Padronização dos casos para utilização no kNN e MLP
caso1 = pd.DataFrame(
    scaler.transform(caso1),
    columns=caso1.columns
)
caso2 = pd.DataFrame(
    scaler.transform(caso2),
    columns=caso2.columns
)

# Demonstração dos casos utilizando o kNN
resultado_knn_caso1 = demonstrar_caso(modelo_knn, caso1, "kNN", 1, y1)
resultado_knn_caso2 = demonstrar_caso(modelo_knn, caso2, "kNN", 2, y2)

# Demonstração dos casos utilizando a Rede Neural MLP
resultado_mlp_caso1 = demonstrar_caso(modelo_mlp, caso1, "MLP", 1, y1)
resultado_mlp_caso2 = demonstrar_caso(modelo_mlp, caso2, "MLP", 2, y2)

# Organização dos resultados do modelo Árvore de Decisão
resultado_tree["Casos de Uso"] = [
    {
        "Nome": "Caso 1",
        "Resultado": resultado_arvore_caso1
    },
    {
        "Nome": "Caso 2",
        "Resultado": resultado_arvore_caso2
    }
]

# Organização dos resultados do kNN
resultado_knn["Casos de Uso"] = [
    {
        "Nome": "Caso 1",
        "Resultado": resultado_knn_caso1
    },
    {
        "Nome": "Caso 2",
        "Resultado": resultado_knn_caso2
    }
]

# Organização dos resultados do MLP
resultado_mlp["Casos de Uso"] = [
    {
        "Nome": "Caso 1",
        "Resultado": resultado_mlp_caso1
    },
    {
        "Nome": "Caso 2",
        "Resultado": resultado_mlp_caso2
    }
]

# Exportação dos resultados gerais e dos casos de uso para um arquivo JSON
resultados = {
    "Caso": {
        "Caso 1":  { 
            "Descricao": "País com grande extensão territorial e participação moderada/alta de eletricidade proveniente de fontes de baixo carbono",
            "Atributos": caso1_original.iloc[0].to_dict(),
        },
        "Caso 2": {
            "Descricao": "País com baixa densidade populacional, emissões de CO₂ moderadas/altas e consumo de energia per capita acima da mediana",
            "Atributos": caso2_original.iloc[0].to_dict(),
        }
    },
    "Modelos": [
        resultado_tree,
        resultado_knn,
        resultado_mlp
    ]
}

with open("resultados.json", "w", encoding="utf-8") as arquivo:
    json.dump(resultados, arquivo, indent=4, ensure_ascii=False)