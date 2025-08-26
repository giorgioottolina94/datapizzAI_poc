# 📚 DatapizzAI Handbook Assistant - Web Interface

Una moderna interfaccia web per l'assistente del manuale SIAE Academy, costruita con FastAPI e vanilla JavaScript.

## 🌟 Caratteristiche Principali

- **💬 Chat Intelligente**: Conversazioni con AI per domande sui contenuti del corso
- **📁 Gestione Indice**: Caricamento e gestione dei documenti indicizzati
- **📓 Notebook Utilities**: Esecuzione e reporting di Jupyter notebooks
- **🔄 Ingestion Multi-Fonte**: 
  - File locali (PDF, DOCX, PPTX, XLSX, ecc.)
  - Google Drive (con autenticazione service account)
  - Discord (messaggi dei canali)
- **📊 Report Statici**: Visualizzazione di report HTML generati
- **🌐 Condivisione Pubblica**: Supporto ngrok per accesso esterno
- **🎨 UI Moderna**: Design responsive e user-friendly

## 🚀 Quick Start

### 1. Installazione Dipendenze

```bash
# Crea virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Configurazione

#### OpenAI API Key
Crea il file `state/openai_key.txt` con la tua chiave API OpenAI:
```bash
echo "your-openai-api-key-here" > demo_minimo/handbook_assistant/state/openai_key.txt
```

#### Google Drive (Opzionale)
1. Crea un service account su Google Cloud Console
2. Scarica il JSON delle credenziali
3. Caricalo tramite l'interfaccia web o salvalo come `state/drive_sa.json`

#### Discord (Opzionale)
1. Crea un bot su Discord Developer Portal
2. Ottieni il token del bot
3. Invita il bot nel server con permessi di lettura messaggi

### 3. Avvio del Server

```bash
cd demo_minimo/handbook_assistant
python -m api.server
```

Il server sarà disponibile su `http://localhost:8000`

### 4. Condivisione Pubblica (Opzionale)

Per condividere il sito pubblicamente:

1. Installa ngrok: `brew install ngrok` (macOS) o scarica da [ngrok.com](https://ngrok.com)
2. Usa il pulsante "🌐 Condividi Pubblicamente" nell'interfaccia web

## 📁 Struttura del Progetto

```
handbook_assistant/
├── api/                    # Backend FastAPI
│   ├── server.py          # Main server application
│   └── services.py        # Business logic
├── web_frontend/          # Frontend statico
│   ├── index.html         # Main HTML page
│   └── assets/
│       ├── app.js         # JavaScript logic
│       └── styles.css     # CSS styling
├── state/                 # File di configurazione
├── data/                  # Dati del corso
├── reports/               # Report HTML generati
└── index/                 # Indice vettoriale
```

## 🔧 API Endpoints

### Chat
- `POST /api/chat` - Invia messaggio alla chat
- `GET /api/history` - Ottieni cronologia chat
- `DELETE /api/history` - Cancella cronologia

### Indice
- `GET /api/index/sources` - Lista fonti indicizzate
- `GET /api/index/summary` - Riassunto dell'indice
- `POST /api/index/clear` - Cancella indice

### Notebooks
- `GET /api/notebooks` - Lista notebooks disponibili
- `POST /api/notebook/run` - Esegui notebook
- `GET /api/notebook/report` - Ottieni report notebook

### Ingestion
- `POST /api/ingest/local` - Avvia ingestion file locali
- `POST /api/ingest/drive` - Avvia ingestion Google Drive
- `POST /api/ingest/discord` - Avvia ingestion Discord
- `GET /api/ingest/state` - Stato ingestion corrente

### Configurazione
- `GET /api/config` - Stato configurazione
- `POST /api/config/openai-key` - Imposta chiave OpenAI
- `POST /api/drive/set-service-account` - Carica credenziali Google Drive

### Utilità
- `POST /api/share/ngrok` - Crea tunnel ngrok pubblico
- `GET /api/preview` - Anteprima file
- `GET /api/reports/list` - Lista report disponibili

## 🎨 Personalizzazione UI

Il design è completamente personalizzabile modificando:

- `web_frontend/assets/styles.css` - Stili e colori
- `web_frontend/assets/app.js` - Comportamento JavaScript
- `web_frontend/index.html` - Struttura HTML

## 🐛 Troubleshooting

### Server non si avvia
- Verifica che la porta 8000 sia libera
- Controlla che tutte le dipendenze siano installate

### Chat non funziona
- Verifica che la chiave OpenAI sia configurata correttamente
- Controlla i log del server per errori

### Google Drive non funziona
- Verifica che il service account JSON sia valido
- Assicurati che le cartelle siano condivise con l'email del service account

### Discord non funziona
- Verifica che il bot abbia i permessi corretti
- Controlla che il token sia valido

## 📝 Licenza

Questo progetto è parte del framework DatapizzAI.

## 🤝 Contributi

Per contribuire al progetto:

1. Fork del repository
2. Crea un branch per la tua feature
3. Commit delle modifiche
4. Push del branch
5. Crea una Pull Request

## 📞 Supporto

Per domande o supporto, apri una issue su GitHub.

---

**Made with ❤️ for SIAE Academy**