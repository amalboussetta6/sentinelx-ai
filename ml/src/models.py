"""Les deux modèles de la baseline."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def create_model(model_name, max_features=0.33, random_state=42):
    """Crée un modèle neuf : "Régression linéaire" ou "Forêt aléatoire" (max_features ne concerne que la forêt)."""
    if model_name == "Régression linéaire":
        return LinearRegression()
    if model_name == "Forêt aléatoire":
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=max_features,
                                     random_state=random_state, n_jobs=-1)
    raise ValueError(f"Modèle inconnu : {model_name}")
