import os
import logging
import asyncio
import signal
import sys
import tracemalloc
from flask import Flask, jsonify
import aiohttp
import talib
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from logging.handlers import RotatingFileHandler  # Import de RotatingFileHandler

# Activer la surveillance de la mémoire
tracemalloc.start()

# Configuration du gestionnaire de logs avec rotation des fichiers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
handler = logging.handlers.RotatingFileHandler('bot_trading.log', maxBytes=5*1024*1024, backupCount=3)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(handler)
logger = logging.getLogger(__name__)
logger.debug("Démarrage de l'application.")

# Variables d'environnement
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
PORT = int(os.getenv("PORT", 8004))

if not DISCORD_WEBHOOK_URL:
    logger.error("La variable d'environnement DISCORD_WEBHOOK_URL est manquante. Veuillez la définir.")
    sys.exit(1)

# Initialisation de Flask
app = Flask(__name__)

# Configuration du gestionnaire de logs pour Flask avec rotation des fichiers
flask_handler = logging.handlers.RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=3)
flask_handler.setLevel(logging.INFO)
flask_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
flask_handler.setFormatter(flask_formatter)
app.logger.addHandler(flask_handler)
app.logger.setLevel(logging.INFO)

# Constantes
CURRENCY = "USD"
CRYPTO_LIST = ["BTC", "ETH", "EUR", "CAD"]
MAX_POSITION_PERCENTAGE = 0.1
CAPITAL = 100
PERFORMANCE_LOG = "trading_performance.csv"
SIGNAL_LOG = "signal_log.csv"

# Récupération des données historiques pour les cryptomonnaies
async def fetch_historical_data(crypto_symbol, currency="USD", interval="hour", limit=2000, max_retries=5, backoff_factor=2):
    logger.debug(f"Début de la récupération des données historiques pour {crypto_symbol}.")
    base_url = "https://min-api.cryptocompare.com/data/v2/"

    # Déterminer le bon endpoint en fonction de l'intervalle
    endpoint = "histohour" if interval == "hour" else "histoday"
    url = f"{base_url}{endpoint}"
    params = {
        "fsym": crypto_symbol.upper(),
        "tsym": currency.upper(),
        "limit": limit,
        "api_key": "799a75ef2ad318c38dfebc92c12723e54e5a650c7eb20159a324db632e35a1b4"
    }

    attempt = 0
    while attempt < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            if data.get("Response") == "Success" and "Data" in data:
                prices = []
                for item in data["Data"].get("Data", []):
                    if all(key in item for key in ["time", "open", "high", "low", "close", "volumeto"]):
                        prices.append({
                            "time": item["time"],
                            "open": item["open"],
                            "high": item["high"],
                            "low": item["low"],
                            "close": item["close"],
                            "volume": item["volumeto"]
                        })

                opens = np.array([item["open"] for item in prices])
                highs = np.array([item["high"] for item in prices])
                lows = np.array([item["low"] for item in prices])
                closes = np.array([item["close"] for item in prices])
                volumes = np.array([item["volume"] for item in prices])

                logger.debug(f"Données récupérées pour {crypto_symbol}: {len(prices)} éléments.")
                return prices, opens, highs, lows, closes, volumes

            else:
                logger.error(f"Erreur API : {data.get('Message', 'Données invalides.')}")
                return [], [], [], [], [], []

        except aiohttp.ClientError as e:
            attempt += 1
            if attempt >= max_retries:
                logger.error(f"Échec après {max_retries} tentatives : {e}")
                return [], [], [], [], [], []
            logger.warning(f"Tentative {attempt}/{max_retries} échouée, nouvelle tentative dans {backoff_factor ** attempt} secondes.")
            await asyncio.sleep(backoff_factor ** attempt)

        except Exception as e:
            logger.error(f"Erreur inattendue : {e}")
            return [], [], [], [], [], []

    logger.error(f"Échec définitif pour {crypto_symbol}.")
    return [], [], [], [], [], []

# Fonction de calcul des indicateurs avec TA-Lib
def calculate_indicators(prices):
    logger.debug("Début du calcul des indicateurs.")
    if len(prices) < 26:
        logger.error("Pas assez de données pour calculer les indicateurs.")
        return {}

    opens = np.array([price["open"] for price in prices])
    highs = np.array([price["high"] for price in prices])
    lows = np.array([price["low"] for price in prices])
    closes = np.array([price["close"] for price in prices])

    sma_short = talib.SMA(closes, timeperiod=10)[-1]
    sma_long = talib.SMA(closes, timeperiod=20)[-1]
    ema_short = talib.EMA(closes, timeperiod=12)[-1]
    ema_long = talib.EMA(closes, timeperiod=26)[-1]
    macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
    upper_band, middle_band, lower_band = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    rsi = talib.RSI(closes, timeperiod=14)[-1]
    slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)

    logger.debug(f"Indicateurs calculés : SMA_short={sma_short}, SMA_long={sma_long}, EMA_short={ema_short}, EMA_long={ema_long}, MACD={macd[-1]}, ATR={atr}, Upper_Band={upper_band[-1]}, Lower_Band={lower_band[-1]}, RSI={rsi}, Stochastic_K={slowk[-1]}, Stochastic_D={slowd[-1]}")
    return {
        "SMA_short": sma_short,
        "SMA_long": sma_long,
        "EMA_short": ema_short,
        "EMA_long": ema_long,
        "MACD": macd[-1],
        "ATR": atr,
        "Upper_Band": upper_band[-1],
        "Lower_Band": lower_band[-1],
        "RSI": rsi,
        "Stochastic_K": slowk[-1],
        "Stochastic_D": slowd[-1],
    }

# Calcul des indicateurs avancés
def calculate_advanced_indicators(prices):
    logger.debug("Début du calcul des indicateurs avancés.")
    if len(prices) < 52:  # L'Ichimoku Cloud nécessite au moins 52 périodes
        logger.error("Pas assez de données pour calculer les indicateurs avancés.")
        return {}

    closes = np.array([price["close"] for price in prices])
    highs = np.array([price["high"] for price in prices])
    lows = np.array([price["low"] for price in prices])

    # Calcul de l'Ichimoku Cloud
    nine_period_high = talib.MAX(highs, timeperiod=9)
    nine_period_low = talib.MIN(lows, timeperiod=9)
    tenkan_sen = (nine_period_high + nine_period_low) / 2

    twenty_six_period_high = talib.MAX(highs, timeperiod=26)
    twenty_six_period_low = talib.MIN(lows, timeperiod=26)
    kijun_sen = (twenty_six_period_high + twenty_six_period_low) / 2

    # Shift operation is not directly supported by TA-Lib, use numpy for shifting arrays
    senkou_span_a = np.roll((tenkan_sen + kijun_sen) / 2, 26)
    fifty_two_period_high = talib.MAX(highs, timeperiod=52)
    fifty_two_period_low = talib.MIN(lows, timeperiod=52)
    senkou_span_b = np.roll((fifty_two_period_high + fifty_two_period_low) / 2, 26)
    chikou_span = np.roll(closes, -26)

    # Calcul du RSI et divergence
    rsi = talib.RSI(closes, timeperiod=14)
    rsi_divergence = closes - rsi  # Exemple simplifié de divergence

    logger.debug(f"Indicateurs avancés calculés : Tenkan-sen={tenkan_sen[-1]}, Kijun-sen={kijun_sen[-1]}, Senkou Span A={senkou_span_a[-1]}, Senkou Span B={senkou_span_b[-1]}, Chikou Span={chikou_span[-1]}, RSI={rsi[-1]}, RSI Divergence={rsi_divergence[-1]}")

    return {
        "Tenkan_sen": tenkan_sen[-1],
        "Kijun_sen": kijun_sen[-1],
        "Senkou_Span_A": senkou_span_a[-1],
        "Senkou_Span_B": senkou_span_b[-1],
        "Chikou_Span": chikou_span[-1],
        "RSI": rsi[-1],
        "RSI_Divergence": rsi_divergence[-1],
    }

# Préparation des données pour le machine learning
def prepare_ml_data(prices):
    closes = np.array([price["close"] for price in prices])
    highs = np.array([price["high"] for price in prices])
    lows = np.array([price["low"] for price in prices])
    volumes = np.array([price["volume"] for price in prices])

    # Créer des features (caractéristiques)
    features = np.column_stack((closes, highs, lows, volumes))
    target = (np.roll(closes, -1) > closes).astype(int)  # 1 si le prix augmente, 0 sinon

    # Normaliser les features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features[:-1], target[:-1]  # Exclure la dernière ligne de target

# Entraînement du modèle de machine learning
def train_ml_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    logger.info(f"Précision du modèle de machine learning : {score}")
    return model

# Prédiction avec le modèle de machine learning
def predict_with_ml(model, features):
    prediction = model.predict(features[-1].reshape(1, -1))
    return "Acheter" if prediction == 1 else "Vendre"

def calculate_sl_tp(entry_price, signal_type, atr, multiplier=1.5):
    logger.debug("Début du calcul des niveaux Stop Loss et Take Profit.")
    if signal_type == "Acheter":
        sl_price = entry_price - (multiplier * atr)
        tp_price = entry_price + (multiplier * atr)
    elif signal_type == "Vendre":
        sl_price = entry_price + (multiplier * atr)
        tp_price = entry_price - (multiplier * atr)
    else:
        logger.error("Type de signal inconnu.")
        return None, None

    logger.debug(f"Stop Loss calculé à : {sl_price}, Take Profit calculé à : {tp_price} (Prix d'entrée : {entry_price})")
    logger.debug("Fin du calcul des niveaux Stop Loss et Take Profit.")
    return sl_price, tp_price

# Analyse des signaux en utilisant les indicateurs avancés et le machine learning
def analyze_signals(prices, model, features):
    logger.debug("Début de l'analyse des signaux avec indicateurs avancés.")
    indicators = calculate_indicators(prices)
    advanced_indicators = calculate_advanced_indicators(prices)

    if not indicators or not advanced_indicators or len(indicators) == 0 or len(advanced_indicators) == 0:
        return "Ne rien faire"

    # Combine existing and new indicators
    if indicators['RSI'] < 30 and advanced_indicators['RSI_Divergence'] > 0:
        decision = "Acheter"
    elif indicators['RSI'] > 70 and advanced_indicators['RSI_Divergence'] < 0:
        decision = "Vendre"
    elif advanced_indicators['Tenkan_sen'] > advanced_indicators['Kijun_sen'] and prices[-1]['close'] > advanced_indicators['Senkou_Span_A']:
        decision = "Acheter"
    elif advanced_indicators['Tenkan_sen'] < advanced_indicators['Kijun_sen'] and prices[-1]['close'] < advanced_indicators['Senkou_Span_B']:
        decision = "Vendre"
    else:
        ml_signal = predict_with_ml(model, features)
        decision = ml_signal

    logger.debug(f"Décision d'action : {decision}")
    return decision

async def send_discord_message(webhook_url, message):
    logger.debug(f"Début de l'envoi d'un message Discord via webhook.")
    data = {
        "content": message
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=data, timeout=10) as response:
                response.raise_for_status()
                response_json = await response.json()
                logger.debug(f"Message envoyé avec succès. Réponse: {response_json}")
    except aiohttp.ClientError as e:
        logger.error(f"Erreur lors de l'envoi du message à Discord : {e}")
    except asyncio.TimeoutError:
        logger.error("La requête a expiré.")
    logger.debug("Fin de l'envoi d'un message Discord.")

def log_memory_usage():
    logger.debug("Début de la journalisation de l'utilisation de la mémoire.")
    current, peak = tracemalloc.get_traced_memory()
    logger.info(f"Utilisation de la mémoire - Actuelle: {current / 10**6} MB, Pic: {peak / 10**6} MB")
    tracemalloc.clear_traces()
    logger.debug("Fin de la journalisation de l'utilisation de la mémoire.")

async def trading_bot():
    logger.info("Début de la tâche de trading.")
    last_sent_signals = {}
    model = None  # Initialiser le modèle de machine learning
    while True:
        logger.info("Début d'une nouvelle itération de trading.")
        try:
            for crypto in CRYPTO_LIST:
                logger.debug(f"Récupération des données historiques pour {crypto}.")
                prices, opens, highs, lows, closes, volumes = await fetch_historical_data(crypto, CURRENCY)
                if prices:
                    logger.debug(f"Données récupérées pour {crypto}: {prices[-1]}")
                    if not model:
                        features, target = prepare_ml_data(prices)
                        model = train_ml_model(features, target)
                    else:
                        features, _ = prepare_ml_data(prices)  # Préparer les nouvelles features pour la prédiction
                    signal = analyze_signals(prices, model, features)
                    logger.debug(f"Signal analysé pour {crypto} avec ML: {signal}")
                    if last_sent_signals.get(crypto) == signal:
                        logger.info(f"Signal déjà envoyé pour {crypto}. Ignoré.")
                        continue
                    last_sent_signals[crypto] = signal
                    entry_price = prices[-1]["close"]
                    atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]  # Calcul de l'ATR
                    sl_price, tp_price = calculate_sl_tp(entry_price, signal, atr)
                    if sl_price is None or tp_price is None:
                        logger.error(f"Erreur dans le calcul des niveaux SL/TP pour {crypto}")
                        continue
                    message = (f"Signal de trading pour {crypto}/{CURRENCY}: {signal}\n"
                               f"Prix d'entrée: {entry_price}\n"
                               f"Stop Loss: {sl_price}\n"
                               f"Take Profit: {tp_price}\n"
                               f"Type : {'Crypto' if crypto in ['BTC', 'ETH'] else 'Fiat'}")
                    logger.debug(f"Envoi du message Discord pour {crypto}: {message}")
                    await send_discord_message(DISCORD_WEBHOOK_URL, message)
                    logger.info(f"Message Discord envoyé pour {crypto}: {signal}")
                else:
                    logger.error(f"Impossible d'analyser les données pour {crypto}, données non disponibles.")
        except Exception as e:
            logger.error(f"Erreur inattendue dans la boucle de trading : {e}")

        # Vérification de la mémoire
        log_memory_usage()
  
        # Attendre avant la prochaine itération
        logger.debug("Attente de 10 minutes avant la prochaine itération.")
        await asyncio.sleep(600)
        logger.debug("Fin de l'attente de 10 minutes.")
    logger.info("Fin de la tâche de trading.")

async def handle_shutdown_signal(signum, frame):
    logger.info(f"Signal d'arrêt reçu : {signum}")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Arrêt propre du bot.")
    sys.exit(0)

def configure_signal_handlers(loop):
    logger.debug("Configuration des gestionnaires de signaux.")
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(handle_shutdown_signal(sig, None)))
    logger.debug("Fin de la configuration des gestionnaires de signaux.")
        
@app.route("/")
def home():
    logger.info("Requête reçue sur '/'")
    return jsonify({"status": "Bot de trading opérationnel."})

async def run_flask():
    logger.debug("Démarrage de l'application Flask.")
    await asyncio.to_thread(app.run, host='0.0.0.0', port=PORT, threaded=True, use_reloader=False, debug=True)
    logger.debug("Fin du démarrage de l'application Flask.")

async def main():
    logger.info("Début de l'exécution principale.")
    await asyncio.gather(
        trading_bot(),
        run_flask()
    )
    logger.info("Fin de l'exécution principale.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exécution interrompue par l'utilisateur.")
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")
    finally:
        logger.info("Arrêt complet.")
