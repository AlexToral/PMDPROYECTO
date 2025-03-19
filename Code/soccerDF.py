import pandas as pd

# Cargar los datasets
soccer_df = pd.read_csv("./Data/SoccerLEagues.csv")
country_facts_df = pd.read_csv("./Data/Country_facts.csv")

# Mostrar las primeras filas de cada dataset para revisión
soccer_df.head(), country_facts_df.head()
