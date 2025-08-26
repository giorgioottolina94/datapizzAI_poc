# 📚 DatapizzAI Handbook Assistant - Web Interface

Una moderna interfaccia web per assistenti handbook intelligenti, costruita con **FastAPI** e vanilla JavaScript, che utilizza estensivamente il **framework DatapizzAI** per tutte le funzionalità di AI e RAG.

## 🎯 **Evoluzione del Progetto**

Inizialmente sviluppato come assistente per il **manuale SIAE Academy**, questo progetto si sta evolvendo in una **soluzione general-purpose** per creare assistenti handbook RAG + agents per qualsiasi dominio:

- 🏫 **SIAE Academy**: Punto di partenza con contenuti educativi e notebook
- 🏢 **Enterprise**: Handbook aziendali, documentazione interna, knowledge base
- 📚 **Documentation**: Guide tecniche, manuali prodotto, FAQ intelligenti  
- 🎓 **Education**: Materiali didattici, corsi online, assistenti tutor
- 🔬 **Research**: Paper collections, knowledge repositories, literature review

### 🚀 **Visione Future**
Il progetto mira a diventare una **piattaforma modulare** dove organizzazioni di qualsiasi tipo possano:
- Caricare la propria documentazione (PDF, Word, Excel, etc.)
- Configurare agenti specializzati per il proprio dominio
- Personalizzare l'interfaccia e i workflow
- Integrare con sistemi esistenti (Slack, Teams, etc.)
- Scalare da uso personale a enterprise

## 🧠 DatapizzAI Framework Integration

Questo progetto è un esempio completo di come utilizzare il framework **DatapizzAI** per creare applicazioni AI avanzate. Utilizza tutti i componenti principali del framework:

### 🤖 **Agents (Agenti AI)**
```python
from datapizzai.agents import Agent
from datapizzai.memory import Memory
from datapizzai.type.type import ROLE, TextBlock

# Creazione dell'agente conversazionale
agent = Agent(
    name="hb-web-agent",
    system_prompt=system_prompt,
    client=chat_client,
    tools=[kb_search_tool, nb_list_tool, nb_run_tool, nb_report_tool],
    memory=memory
)
```

### 💬 **Clients (Client AI)**  
```python
from datapizzai.clients.openai_client import OpenAIClient

# Client per chat (gpt-4o-mini per default - ottimizzato per costi)
chat_client = OpenAIClient(api_key=api_key, model="gpt-4o-mini")

# Client separato per embeddings
embed_client = OpenAIClient(api_key=api_key, model="text-embedding-3-small")
```

### 🔧 **Tools (Strumenti)**
```python
from datapizzai.tools.tools import tool, Tool

@tool(name="kb_search", description="Cerca nei documenti indicizzati")
def kb_search_tool(query: str, mode: str = "hybrid") -> str:
    # Implementazione ricerca semantica con filtri temporali intelligenti
```

### 🧠 **Memory (Memoria Conversazionale)**
```python
from datapizzai.memory import Memory

# Gestione persistente della cronologia delle conversazioni
memory = Memory()
memory.add_message(ROLE.USER, TextBlock(text=user_message))
```

### 📊 **Embeddings & RAG Pipeline**
```python
from datapizzai.embedders.client_embedder import NodeEmbedder
from datapizzai.type.type import Chunk

# Pipeline completa di ingestion e indicizzazione
embedder = NodeEmbedder(client=embed_client)
chunks = [Chunk(id=str(uuid.uuid4()), text=text, embedding=embedding)]
```

## 🌟 Caratteristiche Principali

- **💬 Chat Intelligente**: Conversazioni con AI utilizzando **DatapizzAI Agents** per domande sui contenuti del corso
- **🔍 Ricerca Semantica Avanzata**: RAG implementato con **DatapizzAI embeddings** e ricerca per similarità coseno
- **📁 Gestione Indice**: Caricamento e gestione dei documenti indicizzati tramite **DatapizzAI Chunks**
- **📓 Notebook Utilities**: Esecuzione e reporting di Jupyter notebooks integrati con **DatapizzAI Tools**
- **🔄 Ingestion Multi-Fonte**: 
  - File locali (PDF, DOCX, PPTX, XLSX, ecc.) con parsing automatico
  - Google Drive (con autenticazione service account)
  - Discord (messaggi dei canali) con **DatapizzAI NodeEmbedder**
- **📊 Report Statici**: Visualizzazione di report HTML generati dai notebook
- **🌐 Condivisione Pubblica**: Supporto ngrok per accesso esterno
- **🎨 UI Moderna**: Design responsive e user-friendly con feedback in tempo reale

## 🚀 Quick Start

### 1. Installazione Dipendenze

```bash
# Crea virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Installa dipendenze (include DatapizzAI framework)
pip install -r requirements.txt
```

#### 📦 **Dipendenze DatapizzAI**
Il progetto richiede il framework DatapizzAI e le sue dipendenze:

```txt
# Core DatapizzAI components (installati automaticamente)
datapizzai>=1.0.0           # Framework principale
openai>=1.0.0               # Client OpenAI per chat e embeddings  
anthropic                   # Client Anthropic (opzionale)
google-generativeai         # Client Google (opzionale)
qdrant-client               # Vector database support
sentence-transformers       # Embedding utilities

# Web Framework
fastapi==0.104.1           # API backend
uvicorn[standard]==0.24.0  # ASGI server

# Document Processing (per RAG pipeline)
PyMuPDF                    # PDF parsing
python-docx                # Word documents  
python-pptx                # PowerPoint files
openpyxl                   # Excel files
beautifulsoup4             # HTML parsing
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

## 🏗️ Architettura DatapizzAI

Il progetto implementa un'architettura completa basata sui pattern del framework DatapizzAI:

### 📊 **RAG Pipeline**
1. **Document Ingestion**: Parsing multi-formato (PDF, DOCX, PPTX, XLSX) 
2. **Text Chunking**: Suddivisione intelligente del testo in chunks semantici
3. **Embedding Generation**: Vettorizzazione con `text-embedding-3-small`
4. **Vector Storage**: Indicizzazione in formato JSONL per ricerca rapida
5. **Semantic Search**: Ricerca per similarità coseno con filtri temporali

### 🤖 **Agent Architecture**
```
User Query → Agent → Tools → DatapizzAI Components → Response
    ↓           ↓       ↓            ↓              ↓
  Web UI → hb-web-agent → kb_search → OpenAI Client → Formatted Answer
                     ↓ → nb_list   → File System  → Notebook List  
                     ↓ → nb_run    → Papermill    → Execution Report
                     ↓ → nb_report → HTML Parser  → Report Display
```

### 🔄 **Data Flow**
1. **Ingestion**: `ingest.py` + `discord_ingest.py` → **DatapizzAI Chunks**
2. **Indexing**: **NodeEmbedder** → Vector embeddings → `chunks.jsonl`
3. **Query**: User input → **Agent** → **Tools** → **OpenAI Client**
4. **Retrieval**: Semantic search → Relevant chunks → Context injection
5. **Generation**: **LLM** + Context → Formatted response with citations

### 🧠 **Memory Management**
- **Conversation History**: Persistent storage in `history.json`
- **DatapizzAI Memory**: Automatic context management for multi-turn conversations
- **Session State**: Real-time tracking of ingestion jobs and configurations

## 📁 Struttura del Progetto

```
handbook_assistant/
├── api/                    # Backend FastAPI
│   ├── server.py          # Main server + API endpoints
│   └── services.py        # DatapizzAI integration logic
├── web_frontend/          # Frontend statico
│   ├── index.html         # Main HTML page
│   └── assets/
│       ├── app.js         # JavaScript + real-time updates
│       └── styles.css     # Modern responsive CSS
├── state/                 # Configurazione e memoria
│   ├── history.json       # Chat history (DatapizzAI Memory)
│   ├── openai_key.txt     # API key storage
│   └── drive_sa.json      # Google Drive credentials
├── data/                  # Dati del corso (sorgente)
├── reports/               # Report HTML generati dai notebook
├── index/                 # Indice vettoriale
│   └── chunks.jsonl       # DatapizzAI Chunks con embeddings
├── ingest.py              # Pipeline di ingestion locale
├── discord_ingest.py      # Ingestion Discord con DatapizzAI
└── requirements.txt       # Dipendenze (include DatapizzAI)
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

## 💼 **Casi d'Uso Pratici**

Grazie alla sua architettura modulare basata su **DatapizzAI**, il progetto può essere facilmente adattato per:

### 🏢 **Enterprise & Business**
- **Knowledge Management**: Centralizzare e rendere ricercabile la documentazione aziendale
- **Customer Support**: Assistenti per FAQ, troubleshooting, product knowledge
- **Employee Onboarding**: Guide interattive per nuovi dipendenti
- **Compliance & Policies**: Navigazione intelligente di regolamenti e procedure

### 🎓 **Education & Training**  
- **Course Materials**: Assistenti per materiali didattici e syllabus
- **Research Assistant**: Navigazione di paper, tesi, bibliografia
- **Student Support**: Tutor AI per domande su corsi e assignment
- **Institutional Knowledge**: Archivi storici e documentazione accademica

### 🔬 **Technical & Scientific**
- **API Documentation**: Assistenti per documentazione tecnica
- **Research Papers**: RAG per literature review e knowledge discovery
- **Protocol Management**: Procedure scientifiche e best practices
- **Code Documentation**: Assistenti per codebase e wiki tecniche

## 🎓 Imparare DatapizzAI

Questo progetto è un **esempio pratico completo** di come utilizzare DatapizzAI per creare applicazioni AI avanzate. È perfetto per:

- **Sviluppatori** che vogliono imparare il framework DatapizzAI
- **Data Scientists** interessati a RAG e sistemi conversazionali  
- **Studenti** che studiano architetture AI moderne
- **Aziende** che vogliono implementare assistenti AI interni
- **Organizzazioni** che cercano soluzioni handbook general-purpose

### 📖 **Concetti DatapizzAI Dimostrati**
- ✅ **Agent-based Architecture**: Agenti conversazionali con tools
- ✅ **Multi-Client Support**: OpenAI, Google, Anthropic integration  
- ✅ **RAG Implementation**: Document ingestion, embedding, semantic search
- ✅ **Memory Management**: Persistent conversation history
- ✅ **Tool Integration**: Custom tools per funzionalità specifiche
- ✅ **Streaming Responses**: Real-time AI responses
- ✅ **Multi-modal Processing**: Testo, PDF, immagini, notebook

### 🔗 **Risorse DatapizzAI**
- 📚 **Documentazione**: [docs.datapizza.tech](https://docs.datapizza.tech)
- 🐙 **GitHub**: [DatapizzAI Framework](https://github.com/datapizza-ai/datapizzai)
- 💬 **Community**: Unisciti alla community per supporto e discussioni

## 📝 Licenza

Questo progetto è parte del framework **DatapizzAI** ed è rilasciato sotto licenza MIT.

## 🤝 Contributi

Per contribuire al progetto:

1. Fork del repository
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Commit delle modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push del branch (`git push origin feature/AmazingFeature`)
5. Crea una Pull Request

## 📞 Supporto

Per domande o supporto:
- 🐛 **Bug Reports**: Apri una issue su GitHub
- 💡 **Feature Requests**: Discussioni nelle GitHub Discussions  
- 📖 **Documentazione**: Consulta la documentazione DatapizzAI
- 💬 **Community**: Unisciti alla community DatapizzAI

---

**Made with ❤️ using DatapizzAI Framework for SIAE Academy**
