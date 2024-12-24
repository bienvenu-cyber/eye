# Utiliser une image de base Python plus complète (éviter slim pour une meilleure compatibilité)
FROM python:3.11

# Mettre à jour pip à la dernière version
RUN pip install --upgrade pip

# Installer les dépendances système nécessaires pour psycopg2, TA-Lib et autres outils de compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libpq-dev \
    libtool \
    autoconf \
    libta-lib0-dev \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans l'image Docker
WORKDIR /app

# Copier les fichiers nécessaires dans l'image Docker
COPY requirements.txt /app/requirements.txt
COPY bot.py /app/bot.py

# Définir les variables d'environnement directement dans Dockerfile
ENV TELEGRAM_TOKEN=7635182328:AAEBdBJ3hfUceLUOFZKBHlBMg_yDNDyMIM4
ENV CHAT_ID=7551508160
ENV PORT=8004

# Installer les dépendances Python depuis le fichier requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application écoute
EXPOSE 8004

# Commande de démarrage pour exécuter votre bot
CMD ["python", "bot.py"]