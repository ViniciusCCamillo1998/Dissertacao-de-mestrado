import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt


def infere_values(df, year, target, target_column, main_list, main_path, allow_reduction=False):
    """
    Perform predictive inference for pavement performance indicators using pre-trained XGBoost models.

    This function loads a previously trained model and its corresponding scaler to predict
    the evolution of a given target variable (e.g., IRI, PSI, D0, ATRMED, IDS) for the
    next year based on the historical data of a specific age group. The prediction process
    follows a consistent pipeline: data filtering, scaling, model inference, validation
    against construction events (reinforcement or cut), and conditional assignment of
    predicted values back to the main DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing all data and previous year results.
        year (int): The pavement age (in years) to be used as the base for inference.
        target (str): The model identifier (e.g., "IRI Filtrado (Dados)") used for model selection.
        target_column (str): The column name in `df` containing the current year's measured or inferred values.
        main_list (list[str]): List of input features required by the model.
        main_path (str): Directory path where the scaler and trained XGBoost model are stored.
        allow_reduction (bool, optional): Whether to allow reductions in the target variable
            without maintenance events. Defaults to False.

    Returns:
        pd.DataFrame: Updated DataFrame with predicted values assigned for the following year.

    Raises:
        ValueError: If one or more required columns from `main_list` are missing in `df`.
    """

    print('Iniciando inferencia do ' + target + ' no ano ' + str(year))

    # --- Step 1: Filter Data by Year ---
    # Select only rows corresponding to the specified age for model inference.
    df_temp = df[df['Idade'] == year].copy(deep=True)

    # --- Step 2: Verify Required Columns ---
    missing = [c for c in main_list if c not in df_temp.columns]
    if missing:
        raise ValueError(f"Missing columns in input DataFrame: {missing}")

    print("    ... Lendo Scaler")
    # --- Step 3: Load Scaler for Feature Normalization ---
    scaler_path = os.path.join(main_path, "XGB" + "-" + target + "-training_metrics-scaler_x.pkl")
    scaler = joblib.load(scaler_path)

    print("    ... Lendo Módulo")
    # --- Step 4: Load Trained XGBoost Model ---
    model_path = os.path.join(main_path, "XGB" + "-" + target + "-Model.json")
    model = xgb.Booster()
    model.load_model(model_path)

    print("    ... Transformando dados")
    # --- Step 5: Apply Feature Scaling ---
    X_scaled = scaler.transform(df_temp[main_list])

    print("    ... Inferindo")
    # --- Step 6: Perform Model Inference ---
    dtest = xgb.DMatrix(X_scaled)
    y_pred = model.predict(dtest)

    # --- Step 7: Retrieve Previous Year’s Target Values ---
    y_old = pd.to_numeric(df_temp[target_column], errors='coerce').to_numpy()

    # --- Step 8: Detect Maintenance Events ---
    has_work = (df_temp.get('RESUMO_REFOR.', 0) > 0) | (df_temp.get('RESUMO_CORTE', 0) > 0)

    # --- Step 9: Apply Reduction / Growth Restrictions ---
    # If reduction is not allowed, block unjustified improvement.
    if not allow_reduction:
        if target_column == 'PSI (-1 Ano)':
            # For PSI, block increases without works
            y_final = np.where(has_work.to_numpy(), y_pred, np.minimum(y_pred, y_old))
        else:
            # For other indicators, block decreases without works
            y_final = np.where(has_work.to_numpy(), y_pred, np.maximum(y_pred, y_old))
    else:
        y_final = y_pred

    # --- Step 10: Assign Predictions to Next Year ---
    mask = df["Idade"] == year + 1
    df.loc[mask, target_column] = y_final

    # --- Step 11: Shift Values for -2 Year Columns ---
    if target_column == 'IRI-convertido (-1 Ano)' or target_column == 'PSI (-1 Ano)':
        df.loc[df["Idade"] == year+1, target_column.replace('-1 Ano', '-2 Ano')] = df.loc[df["Idade"] == year, target_column].tolist()
    
    return df


# --- Model Input Lists ---
# Each list defines the feature set required by the corresponding model.
main_list_ATR = [
    'Idade (Dados)', 'PSI (-1 Ano)', 'D0 (-1 Ano)', 'IRI-convertido (-1 Ano)', 'ATRMED (-1 Ano)', 
    'H1ORIGCM (ESTR)', 'H2CM (ESTR)', 'CBRSL (ESTR)', 'NAASHTO acumulado', 'RESUMO_REFOR.', 'RESUMO_CORTE', 
    'RESUMO_PERCTARE', 'RESUMO (-1 Ano)_REFOR.', 'RESUMO (-1 Ano)_CORTE', 'RESUMO (-1 Ano)_PERCTARE'
]

main_list_D0 = [
    'Idade (Dados)', 'PSI (-1 Ano)', 'IDS (-1 Ano)', 'D0 (-1 Ano)', 'IRI-convertido (-1 Ano)', 
    'H1ORIGCM (ESTR)', 'H2CM (ESTR)', 'CBRSL (ESTR)', 'NAASHTO acumulado', 'RESUMO_REFOR.', 
    'RESUMO_CORTE', 'RESUMO_PERCTARE', 'RESUMO (-1 Ano)_REFOR.', 'RESUMO (-1 Ano)_CORTE', 'RESUMO (-1 Ano)_PERCTARE'
]

main_list_IDS = [
    'Idade (Dados)', 'PSI (-1 Ano)', 'D0 (-1 Ano)', 'IRI-convertido (-1 Ano)', 'PSI (-2 Ano)', 'IRI-convertido (-2 Ano)', 
    'H1ORIGCM (ESTR)', 'H2CM (ESTR)', 'CBRSL (ESTR)', 'NAASHTO acumulado', 'RESUMO_REFOR.', 'RESUMO_CORTE', 'RESUMO_PERCTARE', 
    'RESUMO (-1 Ano)_REFOR.', 'RESUMO (-1 Ano)_CORTE', 'RESUMO (-1 Ano)_PERCTARE'
]

main_list_IRI= [
    'Idade (Dados)', 'PSI (-1 Ano)', 'IDS (-1 Ano)', 'D0 (-1 Ano)', 'IRI-convertido (-1 Ano)', 'H1ORIGCM (ESTR)', 
    'H2CM (ESTR)', 'CBRSL (ESTR)', 'NAASHTO acumulado', 'RESUMO_REFOR.', 'RESUMO_CORTE', 'RESUMO_PERCTARE', 
    'RESUMO (-1 Ano)_REFOR.', 'RESUMO (-1 Ano)_CORTE', 'RESUMO (-1 Ano)_PERCTARE'
]

main_list_PSI = [
    'Idade (Dados)', 'PSI (-1 Ano)', 'IDS (-1 Ano)', 'D0 (-1 Ano)', 'IRI-convertido (-1 Ano)', 'H1ORIGCM (ESTR)', 'H2CM (ESTR)', 
    'CBRSL (ESTR)', 'NAASHTO acumulado', 'RESUMO_REFOR.', 'RESUMO_CORTE', 'RESUMO_PERCTARE', 'RESUMO (-1 Ano)_REFOR.', 
    'RESUMO (-1 Ano)_CORTE', 'RESUMO (-1 Ano)_PERCTARE'
]

    
# --- Main Execution Block ---
print("Lendo excel")
df = pd.read_excel("_Dados sinteticos do BD.xlsx")

main_path = "Output dos treinamento"


# --- Sequentially Infer Each Target Across All Years ---
for year in list(range(2, 30)):
    for target in [
        ('ATRMED Filtrado', 'ATRMED (-1 Ano)', main_list_ATR),
        ('D0 Filtrado (Dados)', 'D0 (-1 Ano)', main_list_D0),
        ('IDS Filtrado (Dados)', 'IDS (-1 Ano)', main_list_IDS),
        ('IRI Filtrado (Dados)', 'IRI-convertido (-1 Ano)', main_list_IRI),
        ('PSI Filtrado (Dados)', 'PSI (-1 Ano)', main_list_PSI)
    ]:

        df = infere_values(
            df=df, 
            year=year, 
            target=target[0],
            target_column=target[1],
            main_list=target[2],
            main_path=main_path,
            allow_reduction=True
        )

# --- Export Results ---
df.to_excel('_Dados sinteticos - Processado.xlsx')

# --- Generate Evolution Charts by Group ---
for target in ['PSI (-1 Ano)', 'D0 (-1 Ano)', 'IRI-convertido (-1 Ano)', 'ATRMED (-1 Ano)', 'IDS (-1 Ano)']:
    plt.figure(figsize=(10,6))

    for grupo, dados in df.groupby("Grupo"):
        plt.plot(dados["Idade"], dados[target], marker="o", label=f"Rodovia {grupo}")

    plt.xlabel("Idade (Anos)")
    plt.ylabel(target.replace(" (-1 Ano)", ""))
    plt.title("Evolução por Rodovia")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, max(df['Idade'])+1, 1))
    plt.savefig(target.replace(" (-1 Ano)", "") + '.jpg')

