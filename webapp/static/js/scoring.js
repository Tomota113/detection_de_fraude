/**
 * =============================================================================
 *  scoring.js — Page Scoring Client
 *  Recherche par ID, gauge de probabilité, facteurs explicatifs
 * =============================================================================
 */

// Initialisation au chargement du DOM
document.addEventListener('DOMContentLoaded', () => {
    const btnScore = document.getElementById('btn-score');
    const inputId = document.getElementById('input-client-id');

    if (btnScore) {
        btnScore.addEventListener('click', () => scoreClient());
    }
    if (inputId) {
        inputId.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') scoreClient();
        });

        // Autocomplétion
        inputId.addEventListener('input', debounce(async () => {
            const query = inputId.value.trim();
            if (query.length < 3) {
                hideSuggestions();
                return;
            }
            const results = await apiFetch(`/api/search-clients?q=${encodeURIComponent(query)}`);
            showSuggestions(results || []);
        }, 300));
    }
});

function debounce(fn, delay) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
    };
}

function showSuggestions(clients) {
    let box = document.getElementById('suggestions-box');
    if (!box) {
        box = document.createElement('div');
        box.id = 'suggestions-box';
        box.className = 'suggestions-box';
        const input = document.getElementById('input-client-id');
        input.parentElement.style.position = 'relative';
        input.parentElement.appendChild(box);
    }
    if (clients.length === 0) {
        box.classList.add('hidden');
        return;
    }
    box.innerHTML = clients.map(c =>
        `<div class="suggestion-item" onclick="selectClient('${c}')">${c}</div>`
    ).join('');
    box.classList.remove('hidden');
}

function hideSuggestions() {
    const box = document.getElementById('suggestions-box');
    if (box) box.classList.add('hidden');
}

function selectClient(clientId) {
    document.getElementById('input-client-id').value = clientId;
    hideSuggestions();
    scoreClient();
}

async function scoreClient() {
    const clientId = document.getElementById('input-client-id').value.trim();
    if (!clientId) return;

    hideSuggestions();

    const resultArea = document.getElementById('scoring-result');
    resultArea.classList.remove('hidden');
    resultArea.innerHTML = '<div class="loading-spinner"><div class="spinner"></div>Analyse en cours...</div>';

    const data = await apiFetch(`/api/score/${encodeURIComponent(clientId)}`);

    if (!data || data.error) {
        resultArea.innerHTML = `
            <div class="error-card">
                <div class="error-icon">⚠️</div>
                <div class="error-text">${data?.error || 'Client non trouvé'}</div>
            </div>`;
        return;
    }

    renderScoringResult(data);
}

function renderScoringResult(data) {
    const resultArea = document.getElementById('scoring-result');
    const probaPercent = (data.proba * 100).toFixed(1);

    // Couleur selon le risque
    const riskColors = {
        low: { color: '#059669', bg: 'rgba(5,150,105,0.15)', label: 'Faible' },
        medium: { color: '#D97706', bg: 'rgba(217,119,6,0.15)', label: 'Modéré' },
        high: { color: '#DC2626', bg: 'rgba(220,38,38,0.15)', label: 'Élevé' },
        critical: { color: '#991B1B', bg: 'rgba(153,27,27,0.25)', label: 'Critique' },
    };
    const risk = riskColors[data.risk_level] || riskColors.low;

    // Construire le HTML
    resultArea.innerHTML = `
        <div class="scoring-grid fade-in">
            <!-- Gauge + Risque -->
            <div class="card scoring-main-card">
                <div class="gauge-wrapper">
                    <div class="gauge" id="gauge-display" style="--proba: ${probaPercent}; --color: ${risk.color};">
                        <div class="gauge-inner">
                            <div class="gauge-value">${probaPercent}%</div>
                            <div class="gauge-label">Probabilité de fraude</div>
                        </div>
                    </div>
                    <div class="risk-badge" style="background: ${risk.bg}; color: ${risk.color};">
                        Risque ${risk.label}
                    </div>
                </div>
                <div class="verdict-box" style="border-color: ${risk.color};">
                    ${data.proba >= 0.5
                        ? '🔴 Inspection recommandée — Comportement suspect détecté'
                        : data.proba >= 0.2
                            ? '🟠 Surveillance suggérée — Anomalies mineures'
                            : '🟢 Pas d\'anomalie détectée — Client légitime'
                    }
                </div>
            </div>

            <!-- Profil Client -->
            <div class="card">
                <h3 class="card-title">📋 Profil Client</h3>
                <div class="profile-grid">
                    <div class="profile-item">
                        <span class="profile-label">Client ID</span>
                        <span class="profile-value">${data.profile.client_id}</span>
                    </div>
                    <div class="profile-item">
                        <span class="profile-label">Région</span>
                        <span class="profile-value">${data.profile.region}</span>
                    </div>
                    <div class="profile-item">
                        <span class="profile-label">District</span>
                        <span class="profile-value">${data.profile.district}</span>
                    </div>
                    <div class="profile-item">
                        <span class="profile-label">Catégorie</span>
                        <span class="profile-value">${data.profile.category}</span>
                    </div>
                    <div class="profile-item">
                        <span class="profile-label">Créé le</span>
                        <span class="profile-value">${data.profile.creation_date}</span>
                    </div>
                    <div class="profile-item">
                        <span class="profile-label">Statut réel</span>
                        <span class="profile-value">${data.profile.actual_label === 1 ? '🔴 Fraudeur' : '🟢 Légitime'}</span>
                    </div>
                </div>
            </div>

            <!-- Facteurs -->
            <div class="card">
                <h3 class="card-title">🔍 Facteurs Discriminants</h3>
                <div id="factors-container">
                    ${renderFactors(data.top_factors)}
                </div>
            </div>

            <!-- Historique Consommation -->
            <div class="card">
                <h3 class="card-title">📈 Historique de Consommation</h3>
                <div class="chart-wrapper" style="height: 250px;">
                    <canvas id="chart-history"></canvas>
                </div>
            </div>
        </div>
    `;

    // Rendre la gauge animée via CSS
    requestAnimationFrame(() => {
        const gauge = document.getElementById('gauge-display');
        if (gauge) gauge.classList.add('animated');
    });

    // Graphique historique de consommation
    if (data.history && data.history.length > 0) {
        renderHistoryChart(data.history);
    }
}

function renderFactors(factors) {
    if (!factors || factors.length === 0) return '<p class="text-secondary">Aucun facteur disponible</p>';

    const maxImp = Math.max(...factors.map(f => f.importance));
    return factors.map(f => {
        const width = (f.importance / maxImp * 100).toFixed(1);
        return `
            <div class="factor-bar">
                <span class="factor-name">${f.feature}</span>
                <div class="factor-track">
                    <div class="factor-fill" style="width: ${width}%;"></div>
                </div>
                <span class="factor-value">${f.value}</span>
            </div>`;
    }).join('');
}

function renderHistoryChart(history) {
    destroyChart('chart-history');
    const ctx = document.getElementById('chart-history');
    if (!ctx) return;

    const defaults = getChartDefaults();
    defaults.plugins.legend = { display: false };
    defaults.scales.y.title = {
        display: true, text: 'Consommation',
        color: '#94A3B8', font: { family: 'Inter', size: 11 },
    };

    chartInstances['chart-history'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: history.map(h => h.date),
            datasets: [{
                data: history.map(h => h.consumption),
                borderColor: COLORS.blue,
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 3,
                pointBackgroundColor: COLORS.blue,
                borderWidth: 2,
            }],
        },
        options: defaults,
    });
}
