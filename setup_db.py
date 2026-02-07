import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# 1. Charger les variables d'environnement du fichier .env
# transforme les variables secretes de .env en variable de l'environnement actuel
load_dotenv()

# 2. Récupérer les variables
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
dbname = os.getenv('DB_NAME')

# 3. Créer l'URL de connexion de manière sécurisée
#on crée un url pour se connecter a une base de donnée PostgreSQL 
url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
#se connecte a la base de donnée que tu auras créé au préalable
engine = create_engine(url)

# Test de connexion et envoi du CSV
#try va tester une des commandes si ça ne marche pas on passe a except
#df est une sorte de base de donnée qui s'appelle DataFrame qui se trouve
#dans la ram on appelle ensuite la fonction to_sql qui transforme la dataframe
#en table nommée players dans engine a qui on a donné l'url donc dans notre 
#base de donnée if_exist='replace' pour actualiser notre base a chaque fois 
#qu'on lance ce programme index=false car notre fichier possède deja un index

try:
    df = pd.read_csv('FC26_20250921.csv')
    df.to_sql('players', engine, if_exists='replace', index=False)
    print("Connexion réussie et données envoyées vers PostgreSQL !")
except Exception as e:
    print(f"Erreur : {e}")