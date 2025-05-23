from email.policy import default
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import xgboost
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_recall_curve, f1_score


# print("Dossier courant :", os.getcwd())
# print("\nFichiers présents :", os.listdir())

# Charger le dataset (remplace 'path_to_dataset' par le chemin de ton fichier CSV)
df = pd.read_csv('predictive_maintenance.csv')

cols_exclu = ['UDI', 'Target']
col_num = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in cols_exclu ]

moy = df[col_num].mean().to_list()
ecart_type = df[col_num].std().to_list()

scaler = StandardScaler()
df[col_num] = scaler.fit_transform(df[col_num])

y = df['Target']
X = df.drop(['UDI','Product ID', 'Failure Type', 'Target', 'Type'], axis=1)
X.columns = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_') for col in X.columns]


# Encoder la variable catégorique 'product_quality'
# X = pd.get_dummies(X, columns=['Type'], drop_first=True)


# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


grid_charger = joblib.load('grid_complet.pkl_grid')

model_xgb = grid_charger.best_estimator_

def evaluate(model, nome_model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # cv = ShuffleSplit(5, test_size=0.2)
    cv = StratifiedKFold(5)
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=cv, scoring='f1',
                                               train_sizes=np.linspace(0.1, 1, 10))
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train_score')
    plt.plot(N, val_score.mean(axis=1), label='val_score')
    plt.title(nome_model)
    plt.xlabel('Training Size')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()


# Ajouter après l'entraînement du modèle optimisé (cellule après GridSearchCV)
# Calculer les probabilités pour ajuster le seuil
y_probs = model_xgb.predict_proba(X_test)[:, 1]

# print("Probabilités prédites :")
# print(y_probs[:10])  # Afficher les 10 premières probabilités


# Calculer la courbe précision-rappel
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Trouver le seuil optimal basé sur le F1-score maximum (comme dans ton graphique)
seuils = np.arange(0.0, 1.01, 0.01)
f1_scores = [f1_score(y_test, y_probs > s) for s in seuils]
seuil_optimal = seuils[np.argmax(f1_scores)]
# print(f"Seuil optimal pour F1-score maximum : {seuil_optimal:.2f}")

# Appliquer le seuil optimal
y_pred_final = (y_probs > seuil_optimal).astype(int)

# # Évaluer
# print("Rapport de performance avec seuil optimal :")
# print(classification_report(y_test, y_pred_final))
# print("Matrice de confusion :")
# print(confusion_matrix(y_test, y_pred_final))

# Afficher les résultats
plt.figure(figsize=(10,5))
plt.plot(seuils, f1_scores, label="F1-score", color='blue')
plt.axvline(seuil_optimal, color='red', linestyle='--', label=f"Seuil optimal = {seuil_optimal:.2f}")
plt.xlabel("Seuil de décision")
plt.ylabel("F1-score")
plt.title("Évolution du F1-score selon le seuil de classification")
plt.legend()
plt.grid(True)
plt.close()



# Importance des features
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=model_xgb.feature_importances_, y=feature_names)
plt.title('Importance des variables dans la prédiction des pannes')
plt.xlabel('Importance')
plt.ylabel('Variable')
plt.savefig('importance_variables.png')
plt.close()

# # Sauvegarder le seuil optimal
# joblib.dump(seuil_optimal, 'seuil_optimal.pkl')
# scaler = joblib.load('scaler.pkl')
# model_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
# input_data = {}
# value = [299, 310.4, 1365, 49.1, 226]
# for i in range(len(model_features)):
#     var = model_features[i]
#     input_data[var] = (value[i]) 


# new_data = pd.DataFrame([input_data])
# new_data_scaled = scaler.transform(new_data)

# print(new_data)
# print(new_data_scaled)

# proba = model_xgb.predict_proba(new_data_scaled)[:, 1][0]
# print(f"Probabilité de panne : {proba:.2%}")




# --------------INTERFACE Streamlit-----------




# Charger le modèle, le scaler et le seuil optimal
model_xgb = joblib.load('grid_complet.pkl_grid').best_estimator_
scaler = joblib.load('scaler.pkl')
optimal_threshold = joblib.load('seuil_optimal.pkl')



# Définir les variables utilisées par le modèle
model_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Définir les variables catégoriques
categorical_options = {
    'Type': ['L', 'M', 'H'],
    'Failure Type': ['No Failure', 'Power Failure', 'Tool Wear Failure', 'Overstrain Failure', 'Random Failures', 'Heat Dissipation Failure']
}

# Définir les plages réalistes pour les entrées (basées sur des valeurs typiques ou ton dataset)
numeric_ranges = {
    'Air temperature [K]': {'min': 295.0, 'max': 305.0, 'default': 299.0},
    'Process temperature [K]': {'min': 305.0, 'max': 315.0, 'default': 310.4},
    'Rotational speed [rpm]': {'min': 1000.0, 'max': 3000.0, 'default': 1365.0},
    'Torque [Nm]': {'min': 0.0, 'max': 80.0, 'default': 49.1},
    'Tool wear [min]': {'min': 0.0, 'max': 300.0, 'default': 226.0}
}

# Interface Streamlit
st.title("Prédiction de Pannes d'Équipement Industriel")
st.write(f"Entrez les données de l'équipement pour prédire le risque de panne (seuil optimal : {optimal_threshold:.2f})")

# Fonction pour valider les entrées
def validate_numeric_input(value, var_name, min_val, max_val):
    if not (min_val <= value <= max_val):
        st.error(f"Erreur : {var_name} doit être entre {min_val} et {max_val}.")
        return False
    return True

# Entrées utilisateur
st.sidebar.header("Paramètres de l'équipement")
input_data = {}
# Variables inutiles (affichées mais non utilisées)
udi = st.sidebar.number_input("UDI (Identifiant)", min_value=1, value=1, step=1)
product_id = st.sidebar.text_input("Product ID", value="M14860")

# Variables catégoriques (affichées mais non utilisées dans le modèle final)
type_input = st.sidebar.selectbox("Type", categorical_options['Type'])

default_values = [298.7, 309.8, 1354, 53.3, 212]
i = 0
for var in model_features:
    value = st.sidebar.number_input(
        var,
        min_value=float(numeric_ranges[var]['min']),
        max_value=float(numeric_ranges[var]['max']),
        value=float(default_values[i]) if default_values else float(numeric_ranges[var]['default']),
        step=0.1 if 'temperature' in var or 'Torque' in var else 1.0
    )
    if validate_numeric_input(value, var, numeric_ranges[var]['min'], numeric_ranges[var]['max']):
        input_data[var] = value
        i += 1
        
failure_type_input = st.sidebar.selectbox("Failure Type", categorical_options['Failure Type'])

# Préparer les données pour la prédiction
if input_data:
    new_data = pd.DataFrame([input_data])
    new_data_scaled = scaler.transform(new_data)

    # Prédiction avec seuil optimal
    if st.button("Prédire"):
        proba = model_xgb.predict_proba(new_data_scaled)[:, 1][0]
        prediction = 1 if proba >= optimal_threshold else 0
        st.write(f"**Probabilité de panne : {proba:.2%}**")
        if prediction == 1:
            st.warning("⚠️ **Risque élevé de panne !** Planifiez une maintenance immédiate.")
        else:
            st.success("✅ **Risque faible.** Continuez à surveiller l'équipement.")
        st.write(f"**Décision (seuil {optimal_threshold:.2f}) :** {'Panne détectée' if prediction == 1 else 'Pas de panne'}")

    # Prédiction avec seuil par défaut (0.5)
    if st.button("Prédire avec seuil par défaut (0.5)"):
        proba_default = model_xgb.predict_proba(new_data_scaled)[:, 1][0]
        prediction_default = 1 if proba_default >= 0.5 else 0
        st.write(f"**Probabilité de panne : {proba_default:.2%}**")
        if prediction_default == 1:
            st.warning("⚠️ **Risque élevé de panne !** (seuil 0.5)")
        else:
            st.success("✅ **Risque faible.** (seuil 0.5)")
        st.write(f"**Décision (seuil 0.5) :** {'Panne détectée' if prediction_default == 1 else 'Pas de panne'}")

    # Note explicative
    st.info("**Note :** Le seuil optimal (basé sur le F1-score) est utilisé par défaut pour maximiser la détection des pannes. Le seuil 0.5 est fourni pour comparaison.")
else:
    st.warning("Veuillez corriger les valeurs d'entrée pour continuer.")