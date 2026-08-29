/**
 * =============================================================================
 *  suspects.js — Page Clients Suspects
 *  Tableau trié, filtres, pagination, export CSV
 * =============================================================================
 */

let suspectsLoaded = false;
let currentPage = 0;
const PAGE_SIZE = 25;

async function loadSuspects() {
    if (suspectsLoaded) return;

    // Charger les régions pour le filtre
    const regions = await apiFetch('/api/regions');
    const selectRegion = document.getElementById('select-region');
    if (selectRegion && regions) {
        selectRegion.innerHTML = '<option value="">Toutes les régions</option>' +
            regions.map(r => `<option value="${r}">Région ${r}</option>`).join('');
    }

    // Setup event listeners
    const sliderThreshold = document.getElementById('slider-suspect-threshold');
    const thresholdLabel = document.getElementById('suspect-threshold-value');

    if (sliderThreshold) {
        sliderThreshold.addEventListener('input', () => {
            const val = (sliderThreshold.value / 100).toFixed(2);
            if (thresholdLabel) thresholdLabel.textContent = val;
        });
        sliderThreshold.addEventListener('change', () => {
            currentPage = 0;
            fetchSuspects();
        });
    }

    if (selectRegion) {
        selectRegion.addEventListener('change', () => {
            currentPage = 0;
            fetchSuspects();
        });
    }

    const btnExport = document.getElementById('btn-export-csv');
    if (btnExport) {
        btnExport.addEventListener('click', exportCSV);
    }

    // Charger les données initiales
    await fetchSuspects();
    suspectsLoaded = true;
}

async function fetchSuspects() {
    const threshold = (document.getElementById('slider-suspect-threshold')?.value || 50) / 100;
    const region = document.getElementById('select-region')?.value || '';
    const offset = currentPage * PAGE_SIZE;

    let url = `/api/suspects?threshold=${threshold}&limit=${PAGE_SIZE}&offset=${offset}`;
    if (region) url += `&region=${region}`;

    const data = await apiFetch(url);
    if (!data) return;

    renderSuspectsTable(data);
    renderPagination(data.total);
    renderStats(data);
}

function renderSuspectsTable(data) {
    const tbody = document.getElementById('tbody-suspects');
    if (!tbody) return;

    if (data.suspects.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" style="text-align: center; color: var(--text-secondary); padding: 40px;">
                    Aucun client au-dessus du seuil sélectionné
                </td>
            </tr>`;
        return;
    }

    tbody.innerHTML = data.suspects.map((s, i) => {
        const rank = data.offset + i + 1;
        const probaPercent = (s.proba * 100).toFixed(1);
        const barColor = s.proba >= 0.8 ? COLORS.red
            : s.proba >= 0.5 ? COLORS.orange
            : COLORS.blue;
        const actualBadge = s.actual === 1
            ? '<span class="badge badge-high">Fraudeur</span>'
            : '<span class="badge badge-low">Légitime</span>';

        return `
            <tr class="fade-in" style="animation-delay: ${i * 20}ms;">
                <td class="text-center">${rank}</td>
                <td>
                    <a href="#" class="client-link" onclick="goToScoring('${s.client_id}'); return false;">
                        ${s.client_id}
                    </a>
                </td>
                <td>
                    <div class="proba-cell">
                        <div class="proba-bar-track">
                            <div class="proba-bar-fill" style="width: ${probaPercent}%; background: ${barColor};"></div>
                        </div>
                        <span class="proba-text">${probaPercent}%</span>
                    </div>
                </td>
                <td>${s.region}</td>
                <td>${s.category}</td>
                <td><span class="factor-tag">${s.top_factor}</span></td>
                <td>${actualBadge}</td>
            </tr>`;
    }).join('');
}

function renderPagination(total) {
    const container = document.getElementById('pagination');
    if (!container) return;

    const totalPages = Math.ceil(total / PAGE_SIZE);
    if (totalPages <= 1) {
        container.innerHTML = '';
        return;
    }

    let html = '';

    // Bouton précédent
    html += `<button class="page-btn ${currentPage === 0 ? 'disabled' : ''}"
        onclick="goToPage(${currentPage - 1})" ${currentPage === 0 ? 'disabled' : ''}>←</button>`;

    // Pages
    const maxVisible = 5;
    let start = Math.max(0, currentPage - Math.floor(maxVisible / 2));
    let end = Math.min(totalPages, start + maxVisible);
    if (end - start < maxVisible) start = Math.max(0, end - maxVisible);

    if (start > 0) {
        html += `<button class="page-btn" onclick="goToPage(0)">1</button>`;
        if (start > 1) html += `<span class="page-ellipsis">...</span>`;
    }

    for (let p = start; p < end; p++) {
        html += `<button class="page-btn ${p === currentPage ? 'active' : ''}"
            onclick="goToPage(${p})">${p + 1}</button>`;
    }

    if (end < totalPages) {
        if (end < totalPages - 1) html += `<span class="page-ellipsis">...</span>`;
        html += `<button class="page-btn" onclick="goToPage(${totalPages - 1})">${totalPages}</button>`;
    }

    // Bouton suivant
    html += `<button class="page-btn ${currentPage >= totalPages - 1 ? 'disabled' : ''}"
        onclick="goToPage(${currentPage + 1})" ${currentPage >= totalPages - 1 ? 'disabled' : ''}>→</button>`;

    container.innerHTML = html;
}

function renderStats(data) {
    const stats = document.getElementById('suspect-stats');
    if (!stats) return;
    stats.innerHTML = `
        <strong>${formatNumber(data.total)}</strong> clients au-dessus du seuil
        <strong>${data.threshold}</strong>
    `;
}

function goToPage(page) {
    if (page < 0) return;
    currentPage = page;
    fetchSuspects();
    // Scroll to top of table
    document.getElementById('table-suspects')?.scrollIntoView({ behavior: 'smooth' });
}

function goToScoring(clientId) {
    document.getElementById('input-client-id').value = clientId;
    navigateTo('scoring');
    setTimeout(() => scoreClient(), 100);
}

function exportCSV() {
    const threshold = (document.getElementById('slider-suspect-threshold')?.value || 50) / 100;
    const region = document.getElementById('select-region')?.value || '';

    let url = `/api/suspects/export?threshold=${threshold}`;
    if (region) url += `&region=${region}`;

    // Déclencher le téléchargement
    window.location.href = url;
}
