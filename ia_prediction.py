import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

load_dotenv()
url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(url)

# 1. RÉCUPÉRATION ET NETTOYAGE
query = """
SELECT overall, potential, age, player_positions, nationality_name, league_name, league_level, value_eur 
FROM players 
WHERE value_eur > 0 AND league_name IS NOT NULL AND league_level IS NOT NULL;
"""
df = pd.read_sql(query, engine)

# Nettoyage des Outliers (IQR) pour stabiliser le modèle
Q1 = df['value_eur'].quantile(0.25)
Q3 = df['value_eur'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['value_eur'] >= Q1 - 1.5 * IQR) & (df['value_eur'] <= Q3 + 1.5 * IQR)].copy()

# 2. FEATURE ENGINEERING

# A. Positions
df['player_positions'] = df['player_positions'].str.replace(' ', '')
df_positions = df['player_positions'].str.get_dummies(sep=',')
positions_cols = list(df_positions.columns)

# B. Nationalités (Filtre de crédibilité : 30 joueurs minimum)
counts_nat = df['nationality_name'].value_counts()
reliable_nations = counts_nat[counts_nat >= 30].index

top_nations_rich = (df[df['nationality_name'].isin(reliable_nations)]
                    .groupby('nationality_name')['value_eur']
                    .mean()
                    .nlargest(20)
                    .index)

df['nation_group'] = df['nationality_name'].apply(lambda x: x if x in top_nations_rich else 'Other')
df_nations = pd.get_dummies(df['nation_group'], prefix='nat')
nations_cols = list(df_nations.columns)

# C. Ligues (OPTIMISATION : Basé sur la valeur totale SUM pour le prestige financier)
counts_lg = df['league_name'].value_counts()
reliable_leagues = counts_lg[counts_lg >= 50].index # Seuil monté à 50 pour plus de robustesse

top_leagues_prestige = (df[df['league_name'].isin(reliable_leagues)]
                        .groupby('league_name')['value_eur']
                        .sum() # Somme pour favoriser le poids financier total (Prestige)
                        .nlargest(15) # On se concentre sur le Top 15 mondial
                        .index)

df['league_group'] = df['league_name'].apply(lambda x: x if x in top_leagues_prestige else 'Other')
df_leagues_names = pd.get_dummies(df['league_group'], prefix='lg_name')
leagues_names_cols = list(df_leagues_names.columns)

# Fusion des données transformées
df = pd.concat([df, df_positions, df_nations, df_leagues_names], axis=1)

# Transformations mathématiques
df['value_log'] = np.log(df['value_eur'])
df['overall_exp'] = np.exp(df['overall'] / 100) # Division par 100 pour une courbe exponentielle marquée

# 3. PRÉPARATION IA
# Note : league_level a été retiré des features pour éviter le biais des divisions inférieures
features = ['overall_exp', 'potential', 'age'] + positions_cols + nations_cols + leagues_names_cols
X = df[features]
y = df['value_log']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. ENTRAÎNEMENT
model = LinearRegression()
model.fit(X_train, y_train)

# 5. RÉSULTATS
y_pred_log = model.predict(X_test)
score_r2 = r2_score(y_test, y_pred_log)

# Conversion inverse pour l'erreur en euros
y_test_real = np.exp(y_test)
y_pred_real = np.exp(y_pred_log)

print("\n" + "="*40)
print(f"-> PRÉCISION DU MODÈLE (R²) : {score_r2:.4f}") 
print(f"-> ERREUR MOYENNE : {mean_absolute_error(y_test_real, y_pred_real):,.2f} €")
print("="*40)

print("\n--- TOP 20 DES VARIABLES LES PLUS INFLUENTES ---")
coef_df = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print(coef_df.sort_values(by='Coefficient', ascending=False).head(20))