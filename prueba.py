import joblib

# Carga el archivo problemático
contenido = joblib.load("forecast_col_negocios.pickle")

# Imprime qué es realmente
print("Tipo de dato:", type(contenido))
print("Contenido:", contenido)