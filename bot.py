import os
import requests
import numpy as np
import pandas as pd
import time
import logging
from flask import Flask, jsonify
import asyncio
import signal
import sys
import tracemalloc
import talib
from logging.handlers import RotatingFileHandler
import aiohttp
import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob

# Activer la surveillance de la mémoire
tracemalloc.start()

# Configuration du gestionnaire de logs avec rotation des fichiers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
handler = RotatingFileHandler('bot_trading.log', maxBytes=5*1024*1024, backupCount=3)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(handler)
logger = logging.getLogger(__name__)
logger.debug("Démarrage de l'application.")

# Variables d'environnement
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
PORT = int(os.getenv("PORT", 8002))

if not DISCORD_WEBHOOK_URL:
    logger.error("La variable d'environnement DISCORD_WEBHOOK_URL est manquante. Veuillez la définir.")
    sys.exit(1)

# Initialisation de Flask
app = Flask(__name__)

# Configuration du gestionnaire de logs pour Flask avec rotation des fichiers
flask_handler = RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=3)
flask_handler.setLevel(logging.INFO)
flask_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
flask_handler.setFormatter(flask_formatter)
app.logger.addHandler(flask_handler)
app.logger.setLevel(logging.INFO)

# Constantes
CURRENCY = "USD"
CRYPTO_LIST = ["BTC", "ETH"]
MAX_POSITION_PERCENTAGE = 0.1
CAPITAL = 100
PERFORMANCE_LOG = "trading_performance.csv"
SIGNAL_LOG = "signal_log.csv"

async def fetch_historical_data(crypto_symbol, currency="USD", interval="hour", limit=2000, max_retries=5, backoff_factor=2):
    logger.debug(f"Début de la récupération des données historiques pour {crypto_symbol}.")
    base_url = "https://min-api.cryptocompare.com/data/v2/"
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
                logger.debug(f"Fin de la récupération des données historiques pour {crypto_symbol}.")
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

def calculate_indicators(prices):
    logger.debug("Début du calcul des indicateurs.")
    if len(prices) < 50:
        raise ValueError("Pas assez de données pour calculer les indicateurs.")

    opens = np.array([price["open"] for price in prices])
    highs = np.array([price["high"] for price in prices])
    lows = np.array([price["low"] for price in prices])
    closes = np.array([price["close"] for price in prices])

    sma_short = talib.SMA(closes, timeperiod=10)[-1]
    sma_long = talib.SMA(closes, timeperiod=50)[-1]
    ema_short = talib.EMA(closes, timeperiod=12)[-1]
    ema_long = talib.EMA(closes, timeperiod=26)[-1]
    macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
    upper_band, middle_band, lower_band = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    rsi = talib.RSI(closes, timeperiod=14)[-1]
    slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
    adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1]
    cci = talib.CCI(highs, lows, closes, timeperiod=14)[-1]

    logger.debug(f"Indicateurs calculés : SMA_short={sma_short}, SMA_long={sma_long}, EMA_short={ema_short}, EMA_long={ema_long}, MACD={macd[-1]}, ATR={atr}, Upper_Band={upper_band[-1]}, Lower_Band={lower_band[-1]}, RSI={rsi}, Stochastic_K={slowk[-1]}, Stochastic_D={slowd[-1]}, ADX={adx}, CCI={cci}")
    logger.debug("Fin du calcul des indicateurs.")

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
        "ADX": adx,
        "CCI": cci
    }

async def fetch_sentiment():
    sentiment_score = 0
    try:
        news_api_key = "ctnbgc9r01qn483k4hj0ctnbgc9r01qn483k4hjg"
        news_url = f"https://finnhub.io/api/v1/news?category=general&token={news_api_key}"
        async with aiohttp.ClientSession() as session:
            async with session.get(news_url) as response:
                response.raise_for_status()
                news_data = await response.json()
                articles = news_data.get("articles", [])
                if articles:
                    for article in articles:
                        analysis = TextBlob(article["description"])
                        sentiment_score += analysis.sentiment.polarity
                    sentiment_score /= len(articles)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des nouvelles financières : {e}")
    return sentiment_score

def prepare_data(prices, sentiment_score):
    indicators = calculate_indicators(prices)
    data = pd.DataFrame(indicators, index=[0])
    data['Sentiment'] = sentiment_score
    data['Price'] = [price["close"] for price in prices][-1]
    return data

def train_model(data):
    X = data[['SMA_short', 'SMA_long', 'EMA_short', 'EMA_long', 'MACD', 'ATR', 'Upper_Band', 'Lower_Band', 'RSI', 'Stochastic_K', 'Stochastic_D', 'ADX', 'CCI', 'Sentiment']]
    y = (data['Price'].shift(-1) > data['Price']).astype(int)  # 1 si le prix monte, 0 sinon
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_signals(model, data):
    X = data[['SMA_short', 'SMA_long', 'EMA_short', 'EMA_long', 'MACD', 'ATR', 'Upper_Band', 'Lower_Band', 'RSI', 'Stochastic_K', 'Stochastic_D', 'ADX', 'CCI', 'Sentiment']]
    data['Signal'] = model.predict(X)
    return data

async def analyze_signals(prices):
    logger.debug("Début de l'analyse des signaux.")
    sentiment_score = await fetch_sentiment()
    data = prepare_data(prices, sentiment_score)
    model = train_model(data)
    signals = predict_signals(model, data)
    signal = signals['Signal'].iloc[-1]
    logger.debug(f"Signal prédicté : {signal}")
    return "Acheter" if signal == 1 else "Vendre"

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

async def send_discord_message(webhook_url, message):
    logger.debug(f"Début de l'envoi d'un message Discord via webhook.")
    data = {
        "content": message
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=data, timeout=10) as response:
                response.raise_for_status()
                response_text = await response.text()
                logger.debug(f"Message envoyé avec succès. Réponse: {response_text}")
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
    while True:
        logger.info("Début d'une nouvelle itération de trading.")
        for crypto in CRYPTO_LIST:
            logger.debug(f"Récupération des données historiques pour {crypto}.")
            prices, opens, highs, lows, closes, volumes = await fetch_historical_data(crypto, CURRENCY)
            if prices:
                logger.debug(f"Données récupérées pour {crypto}: {prices[-1]}")
                signal = await analyze_signals(prices)
                if signal:
                    logger.debug(f"Signal analysé pour {crypto}: {signal}")
                    if last_sent_signals.get(crypto) == signal:
                        logger.info(f"Signal déjà envoyé pour {crypto}. Ignoré.")
                        continue
                    last_sent_signals[crypto] = signal
                    entry_price = prices[-1]["close"]
                    atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
                    sl_price, tp_price = calculate_sl_tp(entry_price, signal, atr)
                    if sl_price is None or tp_price is None:
                        logger.error(f"Erreur dans le calcul des niveaux SL/TP pour {crypto}")
                        continue
                    message = (f"Signal de trading pour {crypto}/{CURRENCY}: {signal}\n"
                               f"Prix d'entrée: {entry_price}\n"
                               f"Stop Loss: {sl_price}\n"
                               f"Take Profit: {tp_price}\n")
                    logger.debug(f"Envoi du message Discord pour {crypto}: {message}")
                    await send_discord_message(DISCORD_WEBHOOK_URL, message)
                    logger.info(f"Message Discord envoyé pour {crypto}: {signal}")
                logger.info(f"Signal généré pour {crypto}/{CURRENCY}: {signal}")
            else:
                logger.error(f"Impossible d'analyser les données pour {crypto}, données non disponibles.")

        log_memory_usage()

        logger.debug("Attente de 15 minutes avant la prochaine itération.")
        await asyncio.sleep(900)
        logger.debug("Fin de l'attente de 15 minutes.")
    logger.info("Fin de la tâche de trading.")

async def send_daily_summary(webhook_url):
    logger.debug("Début de l'envoi du résumé journalier sur Discord.")
    
    try:
        df = pd.read_csv(PERFORMANCE_LOG)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        today = datetime.datetime.utcnow().date()
        daily_trades = df[df['timestamp'].dt.date == today]
        
        if not daily_trades.empty:
            summary = daily_trades.to_string(index=False)
            message = f"Résumé des trades du {today}:\n\n{summary}"
        else:
            message = f"Aucun trade effectué le {today}."

        data = {"content": message}
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=data, timeout=10) as response:
                response.raise_for_status()
                response_text = await response.text()
                logger.debug(f"Résumé journalier envoyé avec succès. Réponse: {response_text}")
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi du résumé journalier : {e}")

    logger.debug("Fin de l'envoi du résumé journalier sur Discord.")

scheduler = AsyncIOScheduler()
scheduler.add_job(send_daily_summary, 'interval', days=1, args=[DISCORD_WEBHOOK_URL], next_run_time=datetime.datetime.now() + datetime.timedelta(seconds=10))
scheduler.start()

async def handle_shutdown_signal(signum, frame):
    logger.info(f"Signal d'arrêt reçu : {signum}")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Arrêt propre du bot.")
    sys.exit(
    
    def configure_signal_handlers(loop):
    logger.debug("Configuration des gestionnaires de signaux.")
    for sig in (signal.SIGINT, signal.SIGTERM):  # Gestion des signaux d'interruption
        loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(handle_shutdown_signal(sig, None)))
    logger.debug("Fin de la configuration des gestionnaires de signaux.")
        
# Route de health check
@app.route("/")
def health_check():
    logger.info("Requête reçue sur '/'")
    return "OK", 200

# Route pour vérifier le statut
@app.route("/home")
def home():
    logger.info("Requête reçue sur '/home'")
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