const API_BASE = '';

async function api(path, options = {}) {
  // Show loading bar for chat requests
  if (path === '/api/chat' && options.method === 'POST') {
    showChatLoading();
  }
  
  const res = await fetch(API_BASE + path, {
    method: options.method || 'GET',
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  
  // Hide loading bar for chat requests
  if (path === '/api/chat' && options.method === 'POST') {
    hideChatLoading();
  }
  
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function showChatLoading() {
  // Remove existing loading bar if any
  const existingBar = document.getElementById('chat-loading-bar');
  if (existingBar) {
    existingBar.remove();
  }
  
  // Create new loading bar
  const loadingBar = document.createElement('div');
  loadingBar.id = 'chat-loading-bar';
  loadingBar.className = 'chat-loading-bar active';
  loadingBar.innerHTML = `
    <div class="loading-text">⏳ Elaborazione in corso...</div>
    <div class="loading-progress"></div>
  `;
  
  // Insert after chat input
  const chatInput = document.querySelector('.chat-input');
  if (chatInput && chatInput.parentNode) {
    chatInput.parentNode.insertBefore(loadingBar, chatInput.nextSibling);
  }
}

function hideChatLoading() {
  const loadingBar = document.getElementById('chat-loading-bar');
  if (loadingBar) {
    loadingBar.remove();
  }
}

// Modal preview functionality
function showPreviewModal(items, filename) {
  let modal = document.getElementById('preview-modal');
  if (!modal) {
    modal = el('div', { id: 'preview-modal', class: 'modal' },
      el('div', { class: 'modal-content' },
        el('div', { class: 'modal-header' },
          el('h3', {}, 'Anteprima'),
          el('button', { class: 'modal-close' }, '×')
        ),
        el('div', { id: 'modal-body', class: 'modal-body' })
      )
    );
    document.body.appendChild(modal);
    
    modal.querySelector('.modal-close').addEventListener('click', () => {
      modal.style.display = 'none';
    });
    
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.style.display = 'none';
      }
    });
  }
  
  const modalBody = modal.querySelector('#modal-body');
  modalBody.innerHTML = '';
  
  if (filename) {
    modalBody.appendChild(el('h4', { class: 'preview-filename' }, filename));
  }
  
  if (items.length === 0) {
    modalBody.appendChild(el('div', { class: 'message assistant preview-empty' }, 'ℹ️ Nessun contenuto disponibile per l\'anteprima.'));
  } else {
    items.forEach(item => {
      if (item.kind === 'html_inline') {
        modalBody.appendChild(el('div', { class: 'message assistant', html: item.html }));
      } else if (item.kind === 'markdown' || item.kind === 'text') {
        modalBody.appendChild(el('div', { class: 'message assistant' }, item.text));
      } else if (item.kind === 'code') {
        modalBody.appendChild(el('pre', { class: 'message assistant' }, item.text));
      } else if (item.kind === 'image') {
        const img = new Image();
        img.src = `data:${item.mimetype};base64,${item.data_base64}`;
        img.style.maxWidth = '100%';
        modalBody.appendChild(el('div', { class: 'message assistant' }, img));
      } else if (item.kind === 'info') {
        modalBody.appendChild(el('div', { class: 'message assistant preview-info' }, `ℹ️ ${item.message}`));
      } else if (item.kind === 'error') {
        modalBody.appendChild(el('div', { class: 'message assistant preview-error' }, `❌ ${item.message}`));
      }
    });
  }
  
  modal.style.display = 'block';
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === 'class') e.className = v; else if (k === 'html') e.innerHTML = v; else e.setAttribute(k, v);
  });
  children.forEach(c => {
    if (typeof c === 'string') e.appendChild(document.createTextNode(c)); else if (c) e.appendChild(c);
  });
  return e;
}

function renderMessage(role, content) {
  const div = el('div', { class: `message ${role}` });
  
  // Use Markdown rendering for better formatting
  let html;
  if (typeof marked !== 'undefined') {
    // Configure marked for better code highlighting
    marked.setOptions({
      breaks: true,
      gfm: true
    });
    html = marked.parse(content);
  } else {
    // Fallback if marked is not loaded
    html = content.replace(/\n/g, '<br/>');
  }
  
  const p = el('div', { html: html });
  div.appendChild(p);
  return div;
}

function scrollChatBottom() {
  const h = document.getElementById('chat-history');
  h.scrollTop = h.scrollHeight;
}

function formatFileName(path) {
  const name = path.split('/').pop();
  // Remove common prefixes and format nicely
  return name
    .replace(/_/g, ' ') // Replace underscores with spaces
    .replace(/\.ipynb$/, ' (Notebook)') // Add type indicator for notebooks
    .replace(/\.pdf$/, ' (PDF)')
    .replace(/\.py$/, ' (Python)')
    .replace(/\.md$/, ' (Markdown)');
}

function addSourceItem(path, index) {
  const cont = document.getElementById('preview-items');
  const name = formatFileName(path);
  const row = el('div', { class: 'source-item' },
    el('div', { class: 'source-title' }, `${(index ?? 0) + 1}. ${name}`),
    el('div', { class: 'source-actions' },
      el('button', { class: 'btn btn-ghost' }, 'Apri'),
      el('button', { class: 'btn btn-ghost' }, 'Report AI'),
      el('button', { class: 'btn btn-ghost' }, 'Scarica')
    )
  );
  // Apri anteprima in modal
  row.querySelectorAll('button')[0].addEventListener('click', async () => {
    try {
      const res = await api(`/api/preview?path=${encodeURIComponent(path)}`);
      showPreviewModal(res.items || [], formatFileName(path));
    } catch (e) {
      alert('Errore nel caricamento dell\'anteprima: ' + e.message);
    }
  });
  // Report AI generico (anche per non-notebook)
  row.querySelectorAll('button')[1].addEventListener('click', async () => {
    // placeholder per evitare popup blocker
    const placeholder = window.open('about:blank', '_blank');
    const placeholderOpened = !!placeholder;
    try {
      if (placeholderOpened) {
        try {
          placeholder.document.write('<html><head><title>Generazione report...</title></head><body style="font-family: Inter, ui-sans-serif; padding: 24px; background:#0a0b0f; color:#b8bcc8;"><h3 style="color:#fff; margin:0 0 8px;">Generazione report...</h3><p>Attendere qualche secondo.</p></body></html>');
        } catch (_) {}
      }
      const res = await api('/api/report/ai', {
        method: 'POST',
        body: { path: path, max_chars: 12000 }
      });
      const url = res.url || null;
      if (url) {
        if (placeholderOpened) { try { placeholder.location.href = url; placeholder.focus(); } catch (_) {} }
        else {
          const a = document.createElement('a'); a.href = url; a.target = '_blank'; a.rel = 'noopener'; document.body.appendChild(a); a.click(); a.remove();
        }
      } else {
        alert('Report non generato');
        try { if (placeholderOpened) placeholder.close(); } catch (_) {}
      }
    } catch (e) {
      alert('Errore Report AI: ' + e.message);
      try { if (placeholderOpened) placeholder.close(); } catch (_) {}
    }
  });
  // Scarica file (nuovo endpoint sempre disponibile)
  row.querySelectorAll('button')[2].addEventListener('click', async () => {
    const a = document.createElement('a');
    a.href = `/api/file/download?path=${encodeURIComponent(path)}`;
    a.download = '';
    document.body.appendChild(a);
    a.click();
    a.remove();
  });
  cont.appendChild(row);
}

async function loadReportsList() {
  try {
    const res = await api('/api/reports/list');
    const files = res.items || [];
    const list = document.getElementById('reports-list');
    list.innerHTML = '';
    files.forEach(f => {
      const row = el('div', { class: 'source-item' },
        el('div', { class: 'source-name', title: f.name }, formatFileName(f.name)),
        el('div', { class: 'source-actions' }, el('a', { class: 'btn', href: f.url, target: '_blank' }, 'Apri'))
      );
      list.appendChild(row);
    });
  } catch (e) {
    console.warn('Errore caricamento report', e);
  }
}

// History management functions
let currentHistory = [];

async function loadHistory() {
  const cont = document.getElementById('history-list');
  const stats = document.getElementById('history-stats');
  
  if (cont) {
    cont.innerHTML = '<div class="history-loading">📜 Caricamento cronologia...</div>';
  }
  if (stats) {
    stats.textContent = 'Caricamento...';
  }
  
  try {
    const res = await api('/api/history');
    currentHistory = res.history || [];
    renderHistory();
    updateHistoryStats();
  } catch (e) {
    console.error('Errore caricamento cronologia:', e);
    if (cont) {
      cont.innerHTML = '<div class="message assistant">❌ Errore nel caricamento della cronologia.</div>';
    }
    if (stats) {
      stats.textContent = 'Errore nel caricamento dei dati.';
    }
  }
}

function updateHistoryStats() {
  const stats = document.getElementById('history-stats');
  if (!stats) return;
  
  if (!currentHistory.length) {
    stats.textContent = 'Nessuna conversazione nella cronologia.';
    return;
  }
  
  const userMessages = currentHistory.filter(([role]) => role === 'user').length;
  const assistantMessages = currentHistory.filter(([role]) => role === 'assistant').length;
  stats.textContent = `${currentHistory.length} messaggi totali (${userMessages} domande, ${assistantMessages} risposte)`;
}

function renderHistory() {
  const cont = document.getElementById('history-list');
  if (!cont) return;
  
  const searchInput = document.getElementById('history-search');
  const filterSelect = document.getElementById('history-filter');
  
  const searchTerm = searchInput ? searchInput.value.toLowerCase() : '';
  const filter = filterSelect ? filterSelect.value : 'all';
  
  cont.innerHTML = '';
  
  if (!currentHistory.length) {
    cont.innerHTML = '<div class="message assistant">Nessuna conversazione salvata.</div>';
    return;
  }
  
  let filtered = [...currentHistory];
  
  // Apply role filter
  if (filter !== 'all') {
    filtered = filtered.filter(([role]) => role === filter);
  }
  
  // Apply search filter
  if (searchTerm) {
    filtered = filtered.filter(([role, content]) => 
      content.toLowerCase().includes(searchTerm)
    );
  }
  
  if (!filtered.length) {
    cont.innerHTML = '<div class="message assistant">Nessun messaggio corrisponde ai filtri.</div>';
    return;
  }
  
  // Render messages in reverse order (newest first)
  filtered.reverse().forEach((entry, index) => {
    // Handle both old format [role, content] and new format [role, content, sources]
    const role = entry[0];
    const content = entry[1];
    const sources = entry[2] || []; // Sources may not exist in old format
    
    const messageDiv = el('div', { class: `message ${role}` });
    
    // Add timestamp (for now, show index)
    const timestamp = el('small', { class: 'timestamp' }, `#${filtered.length - index}`);
    
    // Render content with Markdown
    let html;
    if (typeof marked !== 'undefined') {
      marked.setOptions({ breaks: true, gfm: true });
      html = marked.parse(content);
    } else {
      html = content.replace(/\n/g, '<br/>');
    }
    
    const contentDiv = el('div', { html: html });
    messageDiv.appendChild(contentDiv);
    
    // Add sources for assistant messages
    if (role === 'assistant' && sources && sources.length > 0) {
      const sourcesContainer = el('div', { class: 'history-sources' });
      const sourcesHeader = el('div', { class: 'sources-header' }, '📎 Fonti:');
      sourcesContainer.appendChild(sourcesHeader);
      
      sources.forEach(sourcePath => {
        const sourceItem = el('div', { class: 'source-item-history' });
        const sourceName = formatFileName(sourcePath);
        const sourceLink = el('button', { 
          class: 'source-link', 
          title: sourcePath 
        }, sourceName);
        
        sourceLink.addEventListener('click', async () => {
          try {
            const res = await api(`/api/preview?path=${encodeURIComponent(sourcePath)}`);
            showPreviewModal(res.items || [], formatFileName(sourcePath));
          } catch (e) {
            console.error('Error opening source:', e);
            alert('Errore nell\'aprire la fonte: ' + e.message);
          }
        });
        
        sourceItem.appendChild(sourceLink);
        sourcesContainer.appendChild(sourceItem);
      });
      
      messageDiv.appendChild(sourcesContainer);
    }
    
    messageDiv.appendChild(timestamp);
    cont.appendChild(messageDiv);
  });
}

async function clearHistory() {
  if (!confirm('Sei sicuro di voler cancellare tutta la cronologia? Questa azione non può essere annullata.')) {
    return;
  }
  
  try {
    await api('/api/history/clear', { method: 'POST' });
    currentHistory = [];
    renderHistory();
    updateHistoryStats();
    
    // Also clear current chat
    const chatHistory = document.getElementById('chat-history');
    if (chatHistory) {
      chatHistory.innerHTML = '';
    }
    
    alert('Cronologia cancellata con successo.');
  } catch (e) {
    console.error('Errore cancellazione cronologia:', e);
    alert('Errore durante la cancellazione della cronologia.');
  }
}

// Ingestion history management
let currentIngestionHistory = [];

async function loadIngestionHistory() {
  const cont = document.getElementById('ingestion-list');
  const stats = document.getElementById('ingestion-stats');
  
  if (cont) {
    cont.innerHTML = '<div class="history-loading">📋 Caricamento cronologia ingestion...</div>';
  }
  if (stats) {
    stats.textContent = 'Caricamento...';
  }
  
  try {
    const res = await api('/api/ingestion/history');
    currentIngestionHistory = res.history || [];
    renderIngestionHistory();
    updateIngestionStats();
  } catch (e) {
    console.error('Errore caricamento cronologia ingestion:', e);
    if (cont) {
      cont.innerHTML = '<div class="message assistant">❌ Errore nel caricamento della cronologia ingestion.</div>';
    }
    if (stats) {
      stats.textContent = 'Errore nel caricamento dei dati.';
    }
  }
}

function updateIngestionStats() {
  const stats = document.getElementById('ingestion-stats');
  if (!stats) return;
  
  if (!currentIngestionHistory.length) {
    stats.textContent = 'Nessuna ingestion nella cronologia.';
    return;
  }
  
  const completed = currentIngestionHistory.filter(r => r.status === 'completed').length;
  const errors = currentIngestionHistory.filter(r => r.status === 'error').length;
  const inProgress = currentIngestionHistory.filter(r => ['started', 'downloading', 'indexing'].includes(r.status)).length;
  
  stats.textContent = `${currentIngestionHistory.length} operazioni totali (${completed} completate, ${errors} errori, ${inProgress} in corso)`;
}

function renderIngestionHistory() {
  const cont = document.getElementById('ingestion-list');
  if (!cont) return;
  
  const searchInput = document.getElementById('ingestion-search');
  const filterSelect = document.getElementById('ingestion-filter');
  
  const searchTerm = searchInput ? searchInput.value.toLowerCase() : '';
  const filter = filterSelect ? filterSelect.value : 'all';
  
  cont.innerHTML = '';
  
  if (!currentIngestionHistory.length) {
    cont.innerHTML = '<div class="message assistant">Nessuna ingestion salvata.</div>';
    return;
  }
  
  let filtered = [...currentIngestionHistory];
  
  // Apply type filter
  if (filter !== 'all') {
    filtered = filtered.filter(record => record.type === filter);
  }
  
  // Apply search filter
  if (searchTerm) {
    filtered = filtered.filter(record => {
      const searchableText = [
        record.type,
        record.status,
        record.details?.subdir || '',
        ...(record.details?.urls || [])
      ].join(' ').toLowerCase();
      return searchableText.includes(searchTerm);
    });
  }
  
  if (!filtered.length) {
    cont.innerHTML = '<div class="message assistant">Nessuna ingestion corrisponde ai filtri.</div>';
    return;
  }
  
  filtered.forEach((record, index) => {
    const timestamp = new Date(record.timestamp).toLocaleString('it-IT');
    const statusClass = `status-${record.status}`;
    
    const recordDiv = el('div', { class: `ingestion-record ${statusClass}` });
    
    // Meta info (type and status)
    const metaDiv = el('div', { class: 'ingestion-meta' },
      el('span', { class: 'ingestion-type' }, `${getTypeIcon(record.type)} ${record.type.toUpperCase()}`),
      el('span', { class: `ingestion-status ${statusClass}` }, getStatusText(record.status))
    );
    
    // Details
    const details = getIngestionDetails(record);
    const detailsDiv = el('div', { class: 'ingestion-details' }, details);
    
    // Stats
    const stats = getIngestionStatsHtml(record);
    const statsDiv = el('div', { class: 'ingestion-stats', html: stats });
    
    // Timestamp
    const timestampDiv = el('div', { class: 'timestamp' }, `⏰ ${timestamp}`);
    
    recordDiv.appendChild(metaDiv);
    recordDiv.appendChild(detailsDiv);
    if (stats) recordDiv.appendChild(statsDiv);
    recordDiv.appendChild(timestampDiv);
    
    cont.appendChild(recordDiv);
  });
}

function getTypeIcon(type) {
  switch (type) {
    case 'drive': return '☁️';
    case 'local': return '📁';
    case 'discord': return '💬';
    default: return '📄';
  }
}

function getStatusText(status) {
  switch (status) {
    case 'started': return 'Avviato';
    case 'downloading': return 'Download in corso';
    case 'indexing': return 'Indicizzazione';
    case 'completed': return 'Completato';
    case 'error': return 'Errore';
    default: return status;
  }
}

function getIngestionDetails(record) {
  const { type, details, error } = record;
  
  if (error) {
    return `❌ Errore: ${error}`;
  }
  
  switch (type) {
    case 'drive':
      const urlCount = details.url_count || details.urls?.length || 0;
      return `📂 ${urlCount} cartelle Google Drive`;
    case 'local':
      const subdir = details.subdir || 'all';
      return `📁 Cartella locale: ${subdir}`;
    case 'discord':
      return `💬 Canali Discord`;
    default:
      return `Ingestion ${type}`;
  }
}

function getIngestionStatsHtml(record) {
  const stats = [];
  
  if (record.downloaded > 0) {
    stats.push(`<span class="ingestion-stat">⬇️ ${record.downloaded} file scaricati</span>`);
  }
  
  if (record.chunks_before !== undefined && record.chunks_after !== undefined) {
    const added = record.chunks_after - record.chunks_before;
    if (added > 0) {
      stats.push(`<span class="ingestion-stat">➕ +${added} chunks aggiunti</span>`);
    }
    stats.push(`<span class="ingestion-stat">📊 ${record.chunks_before} → ${record.chunks_after} chunks totali</span>`);
  }
  
  return stats.join('');
}

async function openSourcePreview(sourcePath) {
  try {
    // Show loading state
    const historyContainer = document.getElementById('history-list');
    if (!historyContainer) return;
    
    historyContainer.innerHTML = '<div class="message assistant">🔄 Caricamento anteprima...</div>';
    
    const res = await api(`/api/preview?path=${encodeURIComponent(sourcePath)}`);
    const items = res.items || [];
    
    // Clear and setup preview container
    historyContainer.innerHTML = '';
    
    // Add back button and header
    const header = el('div', { class: 'preview-header' });
    
    const backBtn = el('button', {
      class: 'btn btn-ghost preview-back-btn',
      title: 'Torna alla cronologia'
    }, '← Indietro');
    
    backBtn.addEventListener('click', () => {
      renderHistory(); // Go back to history
    });
    
    const titleDiv = el('div', { class: 'preview-title' },
      el('h4', {}, `📄 ${formatFileName(sourcePath)}`),
      el('small', {}, sourcePath)
    );
    
    header.appendChild(backBtn);
    header.appendChild(titleDiv);
    historyContainer.appendChild(header);
    
    // Add content
    const contentContainer = el('div', { class: 'preview-content' });
    
    if (items.length === 0) {
      contentContainer.appendChild(el('div', { 
        class: 'message assistant preview-empty' 
      }, 'ℹ️ Nessun contenuto disponibile per l\'anteprima.'));
    } else {
      items.forEach((item, index) => {
        const itemDiv = el('div', { 
          class: 'preview-item',
          style: index > 0 ? 'margin-top: 16px; border-top: 1px solid var(--border); padding-top: 16px;' : ''
        });

        if (item.kind === 'html_inline') {
          itemDiv.innerHTML = item.html;
          itemDiv.className += ' preview-html';
        } else if (item.kind === 'markdown') {
          if (typeof marked !== 'undefined') {
            marked.setOptions({ breaks: true, gfm: true });
            itemDiv.innerHTML = marked.parse(item.text);
          } else {
            itemDiv.textContent = item.text;
          }
          itemDiv.className += ' preview-markdown';
        } else if (item.kind === 'text') {
          itemDiv.textContent = item.text;
          itemDiv.className += ' preview-text';
        } else if (item.kind === 'code') {
          const pre = el('pre', { class: 'preview-code' });
          pre.textContent = item.text;
          itemDiv.appendChild(pre);
        } else if (item.kind === 'image') {
          const img = new Image();
          img.src = `data:${item.mimetype};base64,${item.data_base64}`;
          img.className = 'preview-image';
          itemDiv.appendChild(img);
          itemDiv.className += ' preview-image-container';
        } else if (item.kind === 'info') {
          itemDiv.textContent = `ℹ️ ${item.message}`;
          itemDiv.className += ' preview-info';
        } else if (item.kind === 'error') {
          itemDiv.textContent = `❌ ${item.message}`;
          itemDiv.className += ' preview-error';
        }

        contentContainer.appendChild(itemDiv);
      });
    }
    
    historyContainer.appendChild(contentContainer);
    
  } catch (e) {
    console.error('Preview error:', e);
    const historyContainer = document.getElementById('history-list');
    if (historyContainer) {
      historyContainer.innerHTML = `
        <div class="preview-header">
          <button class="btn btn-ghost preview-back-btn" onclick="renderHistory()" title="Torna alla cronologia">← Indietro</button>
          <div class="preview-title">
            <h4>❌ Errore</h4>
          </div>
        </div>
        <div class="message assistant preview-error">
          Errore nel caricamento dell'anteprima: ${e.message}
        </div>
      `;
    }
  }
}



async function init() {
  // Chat handlers
  const sendBtn = document.getElementById('chat-send');
  const clearBtn = document.getElementById('chat-clear');
  const input = document.getElementById('chat-text');
  const history = document.getElementById('chat-history');

  const send = async () => {
    const text = input.value.trim();
    if (!text) return;
    
    history.appendChild(renderMessage('user', text));
    input.value = '';
    scrollChatBottom();
    
    try {
      const res = await api('/api/chat', { method: 'POST', body: { text } });
      history.appendChild(renderMessage('assistant', res.reply));
      scrollChatBottom();
      const sources = res.sources || [];
      const cont = document.getElementById('preview-items');
      cont.innerHTML = '';
      sources.forEach((p, i) => addSourceItem(p, i));
      
      // Update history if panel is open
      if (historyPanelOpen) {
        setTimeout(() => loadHistory(), 500); // Small delay to ensure backend is updated
      }
    } catch (e) {
      history.appendChild(renderMessage('assistant', 'Errore: ' + e.message));
      scrollChatBottom();
    }
  };

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
  clearBtn.addEventListener('click', async () => {
    if (confirm('Vuoi cancellare la conversazione corrente? (La cronologia rimarrà salvata)')) {
      history.innerHTML = '';
      document.getElementById('preview-items').innerHTML = '';
    }
  });

  // Notebook management
  let notebookHistoryPanelOpen = false;
  let currentNotebookHistory = [];
  let currentNotebookResults = [];
  
  async function searchNotebooks() {
    console.log('🚀 searchNotebooks() chiamata!');
    
    const query = document.getElementById('nb-query').value.trim();
    const root = document.getElementById('nb-root').value;
    
    console.log('📝 Query:', query, 'Root:', root);
    
    if (!query) {
      console.log('❌ Query vuota, mostro alert');
      alert('Inserisci una query di ricerca');
      return;
    }
    
    console.log('✅ Query valida, iniziando ricerca...');
    
    try {
      // Show loading
      console.log('🔍 Impostando messaggio di loading...');
      const list = document.getElementById('nb-results');
      if (!list) {
        console.error('❌ ERRORE CRITICO: Elemento nb-results non trovato!');
        return;
      }
      list.innerHTML = '<div class="message assistant">🔄 Ricerca notebook in corso...</div>';
      
      console.log('📡 Chiamando API...');
      const res = await api(`/api/notebooks?query=${encodeURIComponent(query)}&root=${encodeURIComponent(root)}&limit=200`);
      console.log('📦 Risposta API:', res);
      
      currentNotebookResults = res.items || [];
      console.log('📊 Risultati salvati:', currentNotebookResults.length, 'items');
      
      // Save to history
      console.log('💾 Salvando nella cronologia...');
      await saveNotebookSearch(query, root, currentNotebookResults.length);
      
      // Render results
      console.log('🎨 Chiamando renderNotebookResults...');
      renderNotebookResults();
      console.log('✅ renderNotebookResults chiamata completata');
      
      // Update history if panel is open
      if (notebookHistoryPanelOpen) {
        console.log('📜 Aggiornando cronologia pannello...');
        setTimeout(() => loadNotebookHistory(), 500);
      }
    } catch (e) {
      console.error('💥 ERRORE in searchNotebooks:', e);
      const list = document.getElementById('nb-results');
      if (list) {
        list.innerHTML = `<div class="message assistant">❌ Errore nella ricerca: ${e.message}</div>`;
      }
    }
  }
  
  function renderNotebookResults() {
    console.log('🔧 renderNotebookResults chiamata con', currentNotebookResults.length, 'risultati');
    console.log('📊 currentNotebookResults:', currentNotebookResults);
    
    const list = document.getElementById('nb-results');
    if (!list) {
      console.error('❌ Elemento nb-results non trovato!');
      return;
    }
    console.log('✅ Elemento nb-results trovato:', list);
    
    list.innerHTML = '';
    console.log('🧹 Lista svuotata');
    
    if (!currentNotebookResults.length) {
      list.innerHTML = '<div class="message assistant">📝 Nessun notebook trovato per questa ricerca.</div>';
      console.log('📝 Mostrato messaggio "nessun risultato"');
      return;
    }
    
    const results = currentNotebookResults.slice(0, 50); // Limit to 50 for performance
    console.log('🎯 Rendering', results.length, 'notebook items');
    console.log('📝 Primi 3 risultati:', results.slice(0, 3));
    
    results.forEach((path, index) => {
      console.log(`🔄 Processing risultato ${index + 1}/${results.length}:`, path);
      
      try {
        const name = formatFileName(path);
        console.log(`📝 Nome formattato:`, name);
        
        const item = el('div', { 
          class: 'notebook-item',
          style: `animation-delay: ${index * 0.05}s;`
        });
        console.log(`📦 Item creato:`, item);
        
        const info = el('div', { class: 'notebook-info' },
          el('div', { class: 'notebook-name', title: name }, name),
          el('div', { class: 'notebook-path', title: path }, path)
        );
        console.log(`ℹ️ Info creata:`, info);
        
        const actions = el('div', { class: 'notebook-actions' },
          el('button', { class: 'btn', title: 'Genera report riassuntivo' }, '📊 Report'),
          el('button', { class: 'btn btn-ghost', title: 'Esegui notebook' }, '🚀 Esegui')
        );
        console.log(`🎯 Actions create:`, actions);
        // Report button
        actions.children[0].addEventListener('click', async () => {
        // Open a placeholder tab immediately to avoid popup blockers
        const placeholder = window.open('about:blank', '_blank');
        const placeholderOpened = !!placeholder;
        try {
          actions.children[0].textContent = '⏳ Generando...';
          actions.children[0].disabled = true;

          // Optional: write a minimal loading page in the placeholder
          if (placeholderOpened) {
            try {
              placeholder.document.write('<html><head><title>Generazione report...</title></head><body style="font-family: Inter, ui-sans-serif; padding: 24px; background:#0a0b0f; color:#b8bcc8;"><h3 style="color:#fff; margin:0 0 8px;">Generazione report...</h3><p>Attendere qualche secondo.</p></body></html>');
            } catch (_) {}
          }

          const detailChars = 12000;
          const res2 = await api('/api/notebook/report', {
            method: 'POST',
            body: { path: path, max_chars: detailChars }
          });

          const url = res2.url || null;
          if (url) {
            if (placeholderOpened) {
              try { placeholder.location.href = url; placeholder.focus(); } catch (_) {}
            } else {
              // Fallback if popup blocked: programmatically click a link
              const a = document.createElement('a');
              a.href = url;
              a.target = '_blank';
              a.rel = 'noopener';
              document.body.appendChild(a);
              a.click();
              a.remove();
            }
          } else {
            throw new Error('Report non generato');
          }
        } catch (e) {
          try { if (placeholderOpened) placeholder.close(); } catch (_) {}
          alert('Errore nella generazione del report: ' + e.message);
        } finally {
          actions.children[0].textContent = '📊 Report';
          actions.children[0].disabled = false;
        }
      });
      
      // Execute button
      actions.children[1].addEventListener('click', async () => {
        try {
          actions.children[1].textContent = '⏳ Eseguendo...';
          actions.children[1].disabled = true;
          
          const res2 = await api('/api/notebook/run', { 
            method: 'POST', 
            body: { path: path } 
          });
          
          if (res2.url) {
            window.open(res2.url, '_blank');
          } else {
            throw new Error('Esecuzione fallita');
          }
        } catch (e) {
          alert('Errore nell\'esecuzione: ' + e.message);
        } finally {
          actions.children[1].textContent = '🚀 Esegui';
          actions.children[1].disabled = false;
        }
        });
        
        console.log(`🔗 Aggiungendo info e actions al item...`);
        item.appendChild(info);
        item.appendChild(actions);
        console.log(`📍 Item completo:`, item);
        console.log(`🔗 Aggiungendo item alla lista...`);
        list.appendChild(item);
        console.log(`✅ Item ${index + 1} aggiunto con successo!`);
        
      } catch (e) {
        console.error(`💥 ERRORE FATALE nel processare risultato ${index + 1}:`, e);
        console.error(`📊 Stack trace:`, e.stack);
        
        // Creo un elemento di fallback molto semplice
        const fallbackItem = document.createElement('div');
        fallbackItem.className = 'notebook-item';
        fallbackItem.innerHTML = `
          <div class="notebook-info">
            <div class="notebook-name">ERRORE: ${path}</div>
            <div class="notebook-path">${e.message}</div>
          </div>
          <div class="notebook-actions">
            <button class="btn" disabled>❌ Errore</button>
          </div>
        `;
        list.appendChild(fallbackItem);
        console.log(`🚨 Aggiunto fallback per item ${index + 1}`);
      }
    });
    
    if (currentNotebookResults.length > 50) {
      const moreInfo = el('div', { class: 'message assistant' }, 
        `📝 Mostrati i primi 50 risultati su ${currentNotebookResults.length} totali.`
      );
      list.appendChild(moreInfo);
      console.log(`ℹ️ Aggiunto messaggio "più risultati"`);
    }
    
    // Verifica finale
    const finalButtons = list.querySelectorAll('button');
    const finalItems = list.querySelectorAll('.notebook-item');
    console.log(`🎯 RISULTATO FINALE:`);
    console.log(`   📦 Items creati: ${finalItems.length}`);
    console.log(`   🔘 Bottoni creati: ${finalButtons.length}`);
    console.log(`   📏 Contenuto HTML (primi 500 caratteri):`, list.innerHTML.substring(0, 500));
    console.log(`✅ renderNotebookResults COMPLETATA!`);
  }
  
  async function saveNotebookSearch(query, root, resultsCount) {
    try {
      await api('/api/notebooks/history', {
        method: 'POST',
        body: {
          query: query,
          root: root,
          results_count: resultsCount,
          timestamp: new Date().toISOString()
        }
      });
    } catch (e) {
      console.error('Error saving notebook search history:', e);
    }
  }
  
  function clearNotebookResults() {
    document.getElementById('nb-results').innerHTML = '';
    currentNotebookResults = [];
  }
  
  function newNotebookSearch() {
    document.getElementById('nb-query').value = '';
    document.getElementById('nb-root').value = 'auto';
    clearNotebookResults();
    document.getElementById('nb-query').focus();
  }
  
  // Notebook event listeners
  console.log('🔌 Registrando event listeners per notebook...');
  
  const nbListBtn = document.getElementById('nb-list');
  const nbQuery = document.getElementById('nb-query');
  const nbClear = document.getElementById('nb-clear');
  const nbNewSearch = document.getElementById('nb-new-search');
  
  if (nbListBtn) {
    console.log('✅ Trovato pulsante nb-list, registrando click listener');
    nbListBtn.addEventListener('click', searchNotebooks);
  } else {
    console.error('❌ ERRORE: Pulsante nb-list non trovato!');
  }
  
  if (nbQuery) {
    console.log('✅ Trovato input nb-query, registrando keydown listener');
    nbQuery.addEventListener('keydown', (e) => {
      console.log('⌨️ Tasto premuto in nb-query:', e.key);
      if (e.key === 'Enter') {
        console.log('🔍 Enter premuto, chiamando searchNotebooks');
        searchNotebooks();
      }
    });
  } else {
    console.error('❌ ERRORE: Input nb-query non trovato!');
  }
  
  if (nbClear) {
    console.log('✅ Trovato pulsante nb-clear, registrando click listener');
    nbClear.addEventListener('click', clearNotebookResults);
  } else {
    console.error('❌ ERRORE: Pulsante nb-clear non trovato!');
  }
  
  if (nbNewSearch) {
    console.log('✅ Trovato pulsante nb-new-search, registrando click listener');
    nbNewSearch.addEventListener('click', newNotebookSearch);
  } else {
    console.error('❌ ERRORE: Pulsante nb-new-search non trovato!');
  }
  
  // Notebook History Panel Management
  function toggleNotebookHistoryPanel() {
    const layout = document.getElementById('notebook-main-layout');
    const panel = document.getElementById('notebook-history-panel');
    
    if (notebookHistoryPanelOpen) {
      // Close panel
      panel.classList.add('closing');
      setTimeout(() => {
        layout.classList.remove('history-expanded');
        panel.style.display = 'none';
        panel.classList.remove('closing');
      }, 400);
      notebookHistoryPanelOpen = false;
    } else {
      // Open panel
      layout.classList.add('history-expanded');
      panel.style.display = 'flex';
      requestAnimationFrame(() => {
        panel.classList.remove('closing');
      });
      notebookHistoryPanelOpen = true;
      loadNotebookHistory();
    }
  }
  
  async function loadNotebookHistory() {
    try {
      const res = await api('/api/notebooks/history');
      currentNotebookHistory = res.history || [];
      renderNotebookHistory();
    } catch (e) {
      console.error('Error loading notebook history:', e);
      currentNotebookHistory = [];
      renderNotebookHistory();
    }
  }
  
  function renderNotebookHistory() {
    const cont = document.getElementById('notebook-history-list');
    if (!cont) return;
    
    const searchInput = document.getElementById('notebook-history-search');
    const filterSelect = document.getElementById('notebook-history-filter');
    
    const searchTerm = searchInput ? searchInput.value.toLowerCase() : '';
    const filter = filterSelect ? filterSelect.value : 'all';
    
    cont.innerHTML = '';
    
    if (!currentNotebookHistory.length) {
      cont.innerHTML = '<div class="message assistant">📝 Nessuna ricerca notebook salvata.</div>';
      return;
    }
    
    let filtered = [...currentNotebookHistory];
    
    // Apply root filter
    if (filter !== 'all') {
      filtered = filtered.filter(record => record.root === filter);
    }
    
    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(record => {
        const searchableText = [
          record.query,
          record.root
        ].join(' ').toLowerCase();
        return searchableText.includes(searchTerm);
      });
    }
    
    if (!filtered.length) {
      cont.innerHTML = '<div class="message assistant">🔍 Nessuna ricerca corrisponde ai filtri.</div>';
      return;
    }
    
    // Sort by timestamp (most recent first)
    filtered.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    
    filtered.forEach((record, index) => {
      const timestamp = new Date(record.timestamp);
      const timeStr = timestamp.toLocaleString('it-IT', {
        day: '2-digit',
        month: '2-digit', 
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
      
      const item = el('div', { class: 'notebook-history-item' });
      
      const query = el('div', { class: 'history-item-query' }, record.query);
      
      const meta = el('div', { class: 'history-item-meta' },
        el('span', { class: 'history-item-root' }, record.root.toUpperCase()),
        el('span', { class: 'history-item-results' }, `${record.results_count || 0} risultati`),
        el('span', { class: 'history-item-time' }, timeStr)
      );
      
      item.appendChild(query);
      item.appendChild(meta);
      
      // Click to rerun search
      item.addEventListener('click', () => {
        document.getElementById('nb-query').value = record.query;
        document.getElementById('nb-root').value = record.root;
        searchNotebooks();
        // Optionally close history panel on mobile
        if (window.innerWidth <= 980) {
          toggleNotebookHistoryPanel();
        }
      });
      
      cont.appendChild(item);
    });
  }
  
  async function clearNotebookHistory() {
    if (confirm('Vuoi cancellare tutta la cronologia delle ricerche notebook? Questa azione non è reversibile.')) {
      try {
        await api('/api/notebooks/history', { method: 'DELETE' });
        currentNotebookHistory = [];
        renderNotebookHistory();
      } catch (e) {
        alert('Errore nella cancellazione della cronologia: ' + e.message);
      }
    }
  }
  
  // Notebook History Event Listeners
  document.getElementById('notebook-history-toggle').addEventListener('click', toggleNotebookHistoryPanel);
  document.getElementById('notebook-history-close').addEventListener('click', toggleNotebookHistoryPanel);
  document.getElementById('notebook-history-refresh').addEventListener('click', loadNotebookHistory);
  document.getElementById('notebook-history-clear-all').addEventListener('click', clearNotebookHistory);
  
  // History search and filter
  const notebookHistorySearch = document.getElementById('notebook-history-search');
  const notebookHistoryFilter = document.getElementById('notebook-history-filter');
  
  if (notebookHistorySearch) {
    notebookHistorySearch.addEventListener('input', renderNotebookHistory);
  }
  
  if (notebookHistoryFilter) {
    notebookHistoryFilter.addEventListener('change', renderNotebookHistory);
  }

  // Ingestion
  document.getElementById('ingest-run').addEventListener('click', async () => {
    const subdir = document.getElementById('ingest-subdir').value || null;
    const status = document.getElementById('local-status') || document.createElement('div');
    if (!document.getElementById('local-status')) {
      status.id = 'local-status';
      document.getElementById('ingest-run').parentNode.appendChild(status);
    }
    
    try {
      status.textContent = 'Avvio ingestion locale...';
      const res = await api('/api/ingest/start-job', { method: 'POST', body: { type: 'local', subdir } });
      const jobId = res.job_id;
      
      // Refresh ingestion history if panel is open
      if (ingestionHistoryPanelOpen) {
        setTimeout(() => loadIngestionHistory(), 1000);
      }
      
      if (!jobId) {
        status.textContent = 'Completato immediatamente';
        return;
      }
      
      // Create progress bar for local ingestion
      status.innerHTML = `
        <div style="margin: 10px 0;">
          <div style="background: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
            <div id="local-progress-bar" style="background: linear-gradient(90deg, #2196F3, #1976D2); height: 100%; width: 0%; transition: width 0.3s ease;"></div>
          </div>
          <div id="local-progress-text" style="margin-top: 8px; font-weight: bold; color: #333;">Indicizzazione file locali...</div>
          <div id="local-progress-details" style="margin-top: 4px; font-size: 0.9em; color: #666;"></div>
        </div>
      `;
      
      const progressBar = document.getElementById('local-progress-bar');
      const progressText = document.getElementById('local-progress-text');
      const progressDetails = document.getElementById('local-progress-details');
      
      let done = false;
      let startTime = Date.now();
      while (!done) {
        await new Promise(r => setTimeout(r, 1500));
        const st = await api(`/api/ingest/state?job_id=${encodeURIComponent(jobId)}`);
        
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const elapsedStr = elapsed > 60 ? `${Math.floor(elapsed/60)}m ${elapsed%60}s` : `${elapsed}s`;
        
        if (st.status === 'error') {
          progressBar.style.background = '#f44336';
          progressBar.style.width = '100%';
          progressText.textContent = `❌ Errore durante l'indicizzazione`;
          progressDetails.textContent = `Dettagli: ${st.error || 'errore sconosciuto'}`;
          break;
        }
        
        if (st.status === 'done') {
          progressBar.style.width = '100%';
          progressText.textContent = `✅ Indicizzazione completata!`;
          progressDetails.textContent = `Chunks: ${st.chunks_before || 0} → ${st.chunks_after || 0} • Tempo: ${elapsedStr}`;
          done = true;
          break;
        }
        
        if (st.status === 'indexing') {
          progressBar.style.width = '60%';
          progressText.textContent = `🔄 Indicizzazione in corso...`;
          progressDetails.textContent = `Tempo trascorso: ${elapsedStr}`;
        }
      }
    } catch (e) {
      status.textContent = 'Errore: ' + e.message;
    }
  });
  document.getElementById('drive-sync').addEventListener('click', async () => {
    const raw = document.getElementById('drive-urls').value || '';
    const urls = raw.split('\n').map(s => s.trim()).filter(Boolean);
    const status = document.getElementById('drive-status');
    status.textContent = 'In corso...';
    try {
      // quick connectivity test
      const test = await api('/api/ingest/drive/test', { method: 'POST', body: { urls } });
      status.textContent = `Accesso OK: ${test.count} file visibili. Avvio indicizzazione...`;
      
      // Refresh ingestion history if panel is open (pre-job)
      if (ingestionHistoryPanelOpen) {
        setTimeout(() => loadIngestionHistory(), 500);
      }

      // Poll progress while il POST è in corso
      let lastCount = 0;
      try {
        const s0 = await api('/api/index/summary');
        lastCount = s0.chunk_count || 0;
      } catch {}
      let stopped = false;
      const timer = setInterval(async () => {
        if (stopped) return;
        try {
          const s = await api('/api/index/summary');
          if (typeof s.chunk_count === 'number' && s.chunk_count !== lastCount) {
            lastCount = s.chunk_count;
            status.textContent = `Indicizzazione in corso... chunks: ${lastCount}`;
          }
        } catch {}
      }, 1500);

      let res;
      try {
        res = await api('/api/ingest/drive', { method: 'POST', body: { urls } });
      } finally {
        stopped = true;
        clearInterval(timer);
      }
      const jobId = res.job_id;
      if (!jobId) {
        const sum = await api('/api/index/summary');
        status.textContent = `Completato. Chunks totali: ${sum.chunk_count}`;
        return;
      }
      // Create progress bar
      status.innerHTML = `
        <div style="margin: 10px 0;">
          <div style="background: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
            <div id="progress-bar" style="background: linear-gradient(90deg, #4CAF50, #45a049); height: 100%; width: 0%; transition: width 0.3s ease;"></div>
          </div>
          <div id="progress-text" style="margin-top: 8px; font-weight: bold; color: #333;">Avvio ingestion...</div>
          <div id="progress-details" style="margin-top: 4px; font-size: 0.9em; color: #666;"></div>
        </div>
      `;
      
      const progressBar = document.getElementById('progress-bar');
      const progressText = document.getElementById('progress-text');
      const progressDetails = document.getElementById('progress-details');
      
      // Poll job state until done
      let done = false;
      let startTime = Date.now();
      while (!done) {
        await new Promise(r => setTimeout(r, 2000));
        const st = await api(`/api/ingest/state?job_id=${encodeURIComponent(jobId)}`);
        
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const elapsedStr = elapsed > 60 ? `${Math.floor(elapsed/60)}m ${elapsed%60}s` : `${elapsed}s`;
        
        if (st.status === 'error') {
          progressBar.style.background = '#f44336';
          progressBar.style.width = '100%';
          progressText.textContent = `❌ Errore durante l'ingestion`;
          progressDetails.textContent = `Dettagli: ${st.error || 'errore sconosciuto'}`;
          break;
        }
        
        if (st.status === 'done') {
          progressBar.style.background = '#4CAF50';
          progressBar.style.width = '100%';
          progressText.textContent = `✅ Ingestion completata!`;
          progressDetails.textContent = `Scaricati ${st.downloaded || 0} file • Chunks: ${st.chunks_before || 0} → ${st.chunks_after || 0} • Tempo: ${elapsedStr}`;
          done = true;
          break;
        }
        
        if (st.status === 'indexing') {
          progressBar.style.width = '75%';
          progressText.textContent = `🔄 Indicizzazione in corso...`;
          progressDetails.textContent = `File scaricati: ${st.downloaded || 0} • Tempo trascorso: ${elapsedStr}`;
        } else if (st.status === 'downloading') {
          const downloaded = st.downloaded || 0;
          const progress = Math.min(50, downloaded > 0 ? (downloaded / 10) * 50 : 10); // Estimate progress
          progressBar.style.width = `${progress}%`;
          progressText.textContent = `⬇️ Download in corso...`;
          progressDetails.textContent = `File scaricati: ${downloaded} • Tempo trascorso: ${elapsedStr}`;
        }
      }
    } catch (e) {
      status.textContent = 'Errore: ' + e.message;
    }
  });
  document.getElementById('discord-backfill').addEventListener('click', async () => {
    const bot_token = document.getElementById('discord-token').value.trim();
    const channel_ids = (document.getElementById('discord-channels').value || '').split(',').map(s => s.trim()).filter(Boolean);
    const max_messages = parseInt(document.getElementById('discord-max').value || '500', 10);
    const statusEl = document.getElementById('discord-status');
    
    if (!bot_token) {
      statusEl.textContent = '⚠️ Bot token richiesto';
      statusEl.style.color = '#ff6b6b';
      return;
    }
    
    if (channel_ids.length === 0) {
      statusEl.textContent = '⚠️ Almeno un Channel ID richiesto';
      statusEl.style.color = '#ff6b6b';
      return;
    }
    
    const btn = document.getElementById('discord-backfill');
    const originalText = btn.textContent;
    btn.textContent = '⏳ Elaborazione...';
    btn.disabled = true;
    statusEl.textContent = `🔄 Scaricando messaggi da ${channel_ids.length} canale/i...`;
    statusEl.style.color = '#4ecdc4';
    
    try {
      const result = await api('/api/ingest/discord', { 
        method: 'POST', 
        body: { bot_token, channel_ids, max_messages } 
      });
      
      if (result.ok) {
        statusEl.textContent = `✅ Completato! ${result.indexed || 0} messaggi indicizzati`;
        statusEl.style.color = '#51cf66';
        // Clear sensitive data
        document.getElementById('discord-token').value = '';
        setTimeout(() => {
          statusEl.textContent = '';
        }, 5000);
      } else {
        statusEl.textContent = '❌ Errore durante il backfill';
        statusEl.style.color = '#ff6b6b';
      }
    } catch (error) {
      console.error('Discord backfill error:', error);
      statusEl.textContent = `❌ Errore: ${error.message || 'Errore sconosciuto'}`;
      statusEl.style.color = '#ff6b6b';
    } finally {
      btn.textContent = originalText;
      btn.disabled = false;
    }
  });

  // Discord help toggle
  document.getElementById('discord-help-toggle').addEventListener('click', () => {
    const helpDiv = document.getElementById('discord-help');
    const toggleBtn = document.getElementById('discord-help-toggle');
    
    if (helpDiv.style.display === 'none') {
      helpDiv.style.display = 'block';
      toggleBtn.textContent = '❌ Nascondi';
    } else {
      helpDiv.style.display = 'none';
      toggleBtn.textContent = '❓ Aiuto';
    }
  });

  // History panel toggle
  let historyPanelOpen = false;
  const historyPanel = document.getElementById('history-panel');
  const chatLayout = document.getElementById('chat-main-layout');
  
  function toggleHistoryPanel() {
    historyPanelOpen = !historyPanelOpen;
    if (historyPanelOpen) {
      historyPanel.style.display = 'flex';
      chatLayout.classList.add('history-expanded');
      document.getElementById('history-toggle').textContent = '📜 Nascondi';
      loadHistory(); // Load history when opened
    } else {
      historyPanel.style.display = 'none';
      chatLayout.classList.remove('history-expanded');
      document.getElementById('history-toggle').textContent = '📜 Cronologia';
    }
  }
  
  // History panel controls
  document.getElementById('history-toggle').addEventListener('click', toggleHistoryPanel);
  document.getElementById('history-close').addEventListener('click', toggleHistoryPanel);
  document.getElementById('history-refresh').addEventListener('click', loadHistory);
  document.getElementById('history-clear-all').addEventListener('click', clearHistory);
  
  // History search and filter
  const historySearch = document.getElementById('history-search');
  const historyFilter = document.getElementById('history-filter');
  historySearch.addEventListener('input', renderHistory);
  historyFilter.addEventListener('change', renderHistory);
  
  // Ingestion history panel toggle
  let ingestionHistoryPanelOpen = false;
  const ingestionHistoryPanel = document.getElementById('ingestion-history-panel');
  const ingestLayout = document.getElementById('ingest-main-layout');
  
  function toggleIngestionHistoryPanel() {
    ingestionHistoryPanelOpen = !ingestionHistoryPanelOpen;
    if (ingestionHistoryPanelOpen) {
      ingestionHistoryPanel.style.display = 'flex';
      ingestLayout.classList.add('history-expanded');
      document.getElementById('ingestion-history-toggle').textContent = '📋 Nascondi';
      loadIngestionHistory(); // Load history when opened
    } else {
      ingestionHistoryPanel.style.display = 'none';
      ingestLayout.classList.remove('history-expanded');
      document.getElementById('ingestion-history-toggle').textContent = '📋 Cronologia';
    }
  }
  
  // Ingestion history controls
  document.getElementById('ingestion-history-toggle').addEventListener('click', toggleIngestionHistoryPanel);
  document.getElementById('ingestion-history-close').addEventListener('click', toggleIngestionHistoryPanel);
  document.getElementById('ingestion-history-refresh').addEventListener('click', loadIngestionHistory);
  
  // Ingestion history search and filter
  const ingestionSearch = document.getElementById('ingestion-search');
  const ingestionFilter = document.getElementById('ingestion-filter');
  ingestionSearch.addEventListener('input', renderIngestionHistory);
  ingestionFilter.addEventListener('change', renderIngestionHistory);

  // Reports list
  loadReportsList();

  // Load config state
  try {
    const cfg = await api('/api/config');
    document.getElementById('cfg-openai-state').textContent = cfg.openai_key_present ? 'Chiave presente' : 'Chiave mancante';
    document.getElementById('cfg-drive-state').textContent = cfg.drive_sa_present ? 'Service account presente' : 'Service account mancante';
  } catch (e) {
    console.warn('Errore lettura config', e);
  }

  // Check for active ingestion job on page load
  checkActiveJob();

  // Save OpenAI key
  document.getElementById('cfg-save-openai').addEventListener('click', async () => {
    const api_key = document.getElementById('cfg-openai-key').value.trim();
    if (!api_key) return alert('Inserisci la chiave');
    await api('/api/config/openai-key', { method: 'POST', body: { api_key } });
    alert('Chiave salvata');
    const cfg = await api('/api/config');
    document.getElementById('cfg-openai-state').textContent = cfg.openai_key_present ? 'Chiave presente' : 'Chiave mancante';
  });

  // Save Drive SA JSON
  document.getElementById('cfg-save-drive').addEventListener('click', async () => {
    const contentRaw = document.getElementById('cfg-drive-json').value;
    try {
      const content = JSON.parse(contentRaw);
      await api('/api/drive/set-service-account', { method: 'POST', body: { content } });
      alert('Service account salvato');
      const cfg = await api('/api/config');
      document.getElementById('cfg-drive-state').textContent = cfg.drive_sa_present ? 'Service account presente' : 'Service account mancante';
    } catch (e) {
      alert('JSON non valido');
    }
  });
}

async function checkActiveJob() {
  try {
    const st = await api('/api/ingest/state');
    if (st.status && st.status !== 'idle' && st.status !== 'done') {
      // There's an active job, show progress bar
      const driveStatus = document.getElementById('drive-status');
      if (driveStatus && st.job_id && st.job_id.startsWith('drive-')) {
        showProgressBar(driveStatus, st.job_id, 'drive');
      }
      
      const localStatus = document.getElementById('local-status');
      if (localStatus && st.job_id && st.job_id.startsWith('local-')) {
        showProgressBar(localStatus, st.job_id, 'local');
      }
    }
  } catch (e) {
    console.warn('Errore controllo job attivo:', e);
  }
}

async function showProgressBar(statusElement, jobId, type) {
  const color = type === 'drive' ? '#4CAF50' : '#2196F3';
  const colorDark = type === 'drive' ? '#45a049' : '#1976D2';
  
  statusElement.innerHTML = `
    <div style="margin: 10px 0;">
      <div style="background: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
        <div id="${type}-progress-bar" style="background: linear-gradient(90deg, ${color}, ${colorDark}); height: 100%; width: 0%; transition: width 0.3s ease;"></div>
      </div>
      <div id="${type}-progress-text" style="margin-top: 8px; font-weight: bold; color: #333;">Ripristino job in corso...</div>
      <div id="${type}-progress-details" style="margin-top: 4px; font-size: 0.9em; color: #666;"></div>
    </div>
  `;
  
  const progressBar = document.getElementById(`${type}-progress-bar`);
  const progressText = document.getElementById(`${type}-progress-text`);
  const progressDetails = document.getElementById(`${type}-progress-details`);
  
  let done = false;
  let startTime = Date.now();
  
  while (!done) {
    await new Promise(r => setTimeout(r, 2000));
    const st = await api(`/api/ingest/state?job_id=${encodeURIComponent(jobId)}`);
    
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const elapsedStr = elapsed > 60 ? `${Math.floor(elapsed/60)}m ${elapsed%60}s` : `${elapsed}s`;
    
    if (st.status === 'error') {
      progressBar.style.background = '#f44336';
      progressBar.style.width = '100%';
      progressText.textContent = `❌ Errore durante l'ingestion`;
      progressDetails.textContent = `Dettagli: ${st.error || 'errore sconosciuto'}`;
      break;
    }
    
    if (st.status === 'done') {
      progressBar.style.width = '100%';
      progressText.textContent = `✅ Ingestion completata!`;
      if (type === 'drive') {
        progressDetails.textContent = `Scaricati ${st.downloaded || 0} file • Chunks: ${st.chunks_before || 0} → ${st.chunks_after || 0} • Tempo: ${elapsedStr}`;
      } else {
        progressDetails.textContent = `Chunks: ${st.chunks_before || 0} → ${st.chunks_after || 0} • Tempo: ${elapsedStr}`;
      }
      done = true;
      break;
    }
    
    if (st.status === 'indexing') {
      progressBar.style.width = type === 'drive' ? '75%' : '60%';
      progressText.textContent = `🔄 Indicizzazione in corso...`;
      if (type === 'drive') {
        progressDetails.textContent = `File scaricati: ${st.downloaded || 0} • Tempo trascorso: ${elapsedStr}`;
      } else {
        progressDetails.textContent = `Tempo trascorso: ${elapsedStr}`;
      }
    } else if (st.status === 'downloading') {
      const downloaded = st.downloaded || 0;
      const progress = Math.min(50, downloaded > 0 ? Math.min(50, (downloaded / 20) * 50) : 10);
      progressBar.style.width = `${progress}%`;
      progressText.textContent = `⬇️ Download in corso...`;
      progressDetails.textContent = `File scaricati: ${downloaded} • Tempo trascorso: ${elapsedStr}`;
    }
  }
}

document.addEventListener('DOMContentLoaded', init);


