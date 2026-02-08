import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

load_dotenv()
url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(url)

# On récupère les colonnes nécessaires pour nos calculs personnalisés
query = """
SELECT overall, potential, age, player_positions, nationality_name, league_level, value_eur 
FROM players 
WHERE value_eur > 0;
"""
df = pd.read_sql(query, engine)
# 1. échelle de position (Score de 1 à 4) cette structure est appelée dictionnaire 
position_weights = {
    'GK': 1.0, 'CB': 2.0, 'RB': 2.2, 'LB': 2.2, 'CDM': 2.8, 
    'CM': 3.0, 'CAM': 3.5, 'RW': 3.8, 'LW': 3.8, 'ST': 4.0
}

def get_pos_score(pos_str):
    positions = [p.strip() for p in pos_str.split(',')]
    scores = [position_weights.get(p, 2.5) for p in positions]
    return sum(scores) / len(scores)

df['position_score'] = df['player_positions'].apply(get_pos_score)

# 2. Dictionnaire de Nations Premium
nations_premium = {
    'England': 1.25, 'Spain': 1.20, 'Brazil': 1.15, 'France': 1.10, 'Argentina': 1.10
}
df['nation_bonus'] = df['nationality_name'].apply(lambda x: nations_premium.get(x, 1.0))

# 3. La puissance de la Ligue (Inverser le league_level)
# Un level 1 (D1) devient 4, un level 4 (D4) devient 1.
df['league_strength'] = 5 - df['league_level']

# Sélection des colonnes finales pour l'IA
features = ['overall', 'potential', 'age', 'position_score', 'nation_bonus', 'league_strength']
X = df[features]
y = df['value_eur']

# Découpage 80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ÉTAPE 4 : CALCULS ET PRÉDICTIONS ---
model = LinearRegression()
model.fit(X_train, y_train)

# On génère les prédictions
y_pred = model.predict(X_test)

# --- ÉTAPE 5 : AFFICHAGE SÉCURISÉ ---
print("\n" + "="*40)
print("       RAPPORT DE PERFORMANCE IA")
print("="*40)

# On calcule le R2
score_r2 = r2_score(y_test, y_pred)
print(f"-> PRÉCISION DU MODÈLE (R²) : {score_r2:.4f}") 

# On nettoie l'affichage des chiffres (Format Monétaire)
pd.options.display.float_format = '{:,.2f}'.format

print("\n--- IMPACT DÉTAILLÉ DES VARIABLES ---")
coef_df = pd.DataFrame(model.coef_, features, columns=['Valeur en €'])
print(coef_df)
print("="*40)
print("Fin de l'exécution du script.")