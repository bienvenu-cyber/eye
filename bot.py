import os
import numpy as np
import pandas as pd
import time
from flask import Flask
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from telegram import Bot
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

# Charger les variables d'environnement
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # Récupère le token Telegram depuis les variables d'environnement
CHAT_ID = os.getenv("CHAT_ID")  # Récupère le chat ID depuis les variables d'environnement

# Initialisation du bot Telegram
bot = Bot(token=TELEGRAM_TOKEN)

# Liste des cryptomonnaies à surveiller
CRYPTO_LIST = ["BTC-USD", "ETH-USD", "ADA-USD"]  # Utiliser les tickers de yfinance

# Fichier de suivi des performances
PERFORMANCE_LOG = "trading_performance.csv"

# Fonction pour récupérer les données historiques avec yfinance
def fetch_crypto_data(crypto_id, period="1y"):
    data = yf.download(crypto_id, period=period)
    return data['Close'].values

# Fonction pour entraîner un modèle de machine learning (à améliorer)
def train_ml_model(data, target):
    # Division des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Modèle de régression logistique (à remplacer par un modèle plus complexe si nécessaire)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model

# Fonction pour calculer les indicateurs techniques
def calculate_indicators(prices):
    # Calculer des indicateurs plus complets (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
    sma_short = prices[-10:].mean()
    sma_long = prices[-20:].mean()

    return sma_short, sma_long  # Ajouter d'autres indicateurs ici

# Fonction pour analyser les signaux avec le modèle ML
def analyze_signals(prices, model):
    # Calculer les indicateurs
    indicators = calculate_indicators(prices)

    # Préparer les données pour le modèle
    features = np.array(indicators).reshape(1, -1)
    prediction = model.predict(features)

    # Signal basé sur le modèle ML
    buy_signal = prediction[0] == 1

    return buy_signal  # Vous pouvez aussi retourner stop_loss et take_profit ici

# Fonction principale pour analyser une crypto
def analyze_crypto(crypto, model):
    prices = fetch_crypto_data(crypto)
    if prices is not None:
        buy_signal = analyze_signals(prices, model)
        # Envoyer un message Telegram avec le signal
        if buy_signal:
            bot.send_message(chat_id=CHAT_ID, text=f"Signal d'achat pour {crypto}!")
        else:
            bot.send_message(chat_id=CHAT_ID, text=f"Pas de signal d'achat pour {crypto}.")

# Fonction Flask pour exposer l'API
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot de trading en cours d'exécution!"

@app.route("/start_bot")
def start_bot():
    data = fetch_crypto_data("BTC-USD", "5y")
    features = calculate_indicators(data)
    targets = np.random.choice([0, 1], size=len(features))  # Ceci est un exemple, vous devrez définir votre stratégie

    # Entraîner le modèle
    model = train_ml_model(features, targets)

    # Lancer l'analyse des cryptomonnaies
    while True:
        with ThreadPoolExecutor() as executor:
            executor.map(lambda crypto: analyze_crypto(crypto, model), CRYPTO_LIST)
        time.sleep(300)  # Attendre 5 minutes avant de vérifier à nouveau

    return "Bot lancé en arrière-plan!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))  # Le port est récupéré de l'environnement