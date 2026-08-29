/**
 * =============================================================================
 *  performance.js — Page Performance du Modèle
 *  ROC, PR, Confusion, Radar, Feature Importance, Seuil interactif
 * =============================================================================
 */

let performanceLoaded = false;

async function loadPerformance() {
    if (performanceLoaded) return;

    // Charger tout en parallèle
    const [roc, pr, confusion, cv, fi] = await Promise.all([
        apiFetch('/api/model/roc'),
        apiFetch('/api/model/pr'),
        apiFetch('/api/model/confusion?threshold=0.5'),
        apiFetch('/api/model/cv-results'),
        apiFetch('/api/feature-importance?n_top=20'),
    ]);

    // --- KPI Cards ---
    if (roc) {
        const kpiRoc = document.querySelector('#kpi-roc .kpi-value');
        if (kpiRoc) kpiRoc.textContent = roc.auc;
    }
    if (pr) {
        const kpiPr = document.querySelector('#kpi-pr .kpi-value');
        if (kpiPr) kpiPr.textContent = pr.ap;
    }
    if (confusion) {
        const kpiF1 = document.querySelector('#kpi-f1 .kpi-value');
        const kpiAcc = document.querySelector('#kpi-accuracy .kpi-value');
        if (kpiF1) kpiF1.textContent = confusion.metrics.f1;
        if (kpiAcc) kpiAcc.textContent = confusion.metrics.accuracy;
    }

    // --- Courbe ROC ---
    if (roc) renderRocChart(roc);

    // --- Courbe PR ---
    if (pr) renderPrChart(pr);

    // --- Matrice de Confusion ---
    if (confusion) renderConfusionMatrix(confusion);

    // --- Radar des modèles ---
    if (cv) renderRadarChart(cv);

    // --- Feature Importance ---
    if (fi) renderFeatureImportance(fi);

    // --- Curseur de seuil ---
    setupThresholdSlider();

    performanceLoaded = true;
}

function renderRocChart(roc) {
    destroyChart('chart-roc');
    const ctx = document.getElementById('chart-roc');
    if (!ctx) return;

    const defaults = getChartDefaults();
    defaults.plugins.legend = {
        labels: { color: '#94A3B8', font: { family: 'Inter' } },
    };
    defaults.scales.x.title = {
        display: true, text: 'Taux de Faux Positifs (FPR)',
        color: '#94A3B8', font: { family: 'Inter', size: 12 },
    };
    defaults.scales.y.title = {
        display: true, text: 'Taux de Vrais Positifs (TPR)',
        color: '#94A3B8', font: { family: 'Inter', size: 12 },
    };
    defaults.scales.x.min = 0; defaults.scales.x.max = 1;
    defaults.scales.y.min = 0; defaults.scales.y.max = 1;

    chartInstances['chart-roc'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: roc.fpr,
            datasets: [
                {
                    label: `LightGBM (AUC = ${roc.auc})`,
                    data: roc.tpr,
                    borderColor: COLORS.blue,
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true,
                    tension: 0.2,
                    pointRadius: 0,
                    borderWidth: 2.5,
                },
                {
                    label: 'Aléatoire (AUC = 0.5)',
                    data: roc.fpr.map((_, i) => roc.fpr[i]),
                    borderColor: '#6B7280',
                    borderDash: [6, 4],
                    pointRadius: 0,
                    borderWidth: 1.5,
                    fill: false,
                },
            ],
        },
        options: defaults,
    });
}

function renderPrChart(pr) {
    destroyChart('chart-pr');
    const ctx = document.getElementById('chart-pr');
    if (!ctx) return;

    const defaults = getChartDefaults();
    defaults.plugins.legend = {
        labels: { color: '#94A3B8', font: { family: 'Inter' } },
    };
    defaults.scales.x.title = {
        display: true, text: 'Rappel (Recall)',
        color: '#94A3B8', font: { family: 'Inter', size: 12 },
    };
    defaults.scales.y.title = {
        display: true, text: 'Précision',
        color: '#94A3B8', font: { family: 'Inter', size: 12 },
    };
    defaults.scales.x.min = 0; defaults.scales.x.max = 1;
    defaults.scales.y.min = 0; defaults.scales.y.max = 1;

    chartInstances['chart-pr'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: pr.recall,
            datasets: [
                {
                    label: `LightGBM (AP = ${pr.ap})`,
                    data: pr.precision,
                    borderColor: COLORS.green,
                    backgroundColor: 'rgba(5, 150, 105, 0.1)',
                    fill: true,
                    tension: 0.2,
                    pointRadius: 0,
                    borderWidth: 2.5,
                },
            ],
        },
        options: defaults,
    });
}

function renderConfusionMatrix(data) {
    const container = document.getElementById('confusion-matrix');
    if (!container) return;

    const cm = data.matrix;
    const total = cm.tn + cm.fp + cm.fn + cm.tp;

    container.innerHTML = `
        <div class="cm-grid">
            <div class="cm-cell cm-tn">
                <div class="cm-count">${formatNumber(cm.tn)}</div>
                <div class="cm-pct">${(cm.tn / total * 100).toFixed(1)}%</div>
                <div class="cm-label">Vrai Négatif</div>
            </div>
            <div class="cm-cell cm-fp">
                <div class="cm-count">${formatNumber(cm.fp)}</div>
                <div class="cm-pct">${(cm.fp / total * 100).toFixed(1)}%</div>
                <div class="cm-label">Faux Positif</div>
            </div>
            <div class="cm-cell cm-fn">
                <div class="cm-count">${formatNumber(cm.fn)}</div>
                <div class="cm-pct">${(cm.fn / total * 100).toFixed(1)}%</div>
                <div class="cm-label">Faux Négatif</div>
            </div>
            <div class="cm-cell cm-tp">
                <div class="cm-count">${formatNumber(cm.tp)}</div>
                <div class="cm-pct">${(cm.tp / total * 100).toFixed(1)}%</div>
                <div class="cm-label">Vrai Positif</div>
            </div>
        </div>
        <div class="cm-metrics">
            <span>Précision: <b>${data.metrics.precision}</b></span>
            <span>Rappel: <b>${data.metrics.recall}</b></span>
            <span>F1: <b>${data.metrics.f1}</b></span>
        </div>
    `;
}

function renderRadarChart(cv) {
    destroyChart('chart-radar');
    const ctx = document.getElementById('chart-radar');
    if (!ctx) return;

    const models = Object.keys(cv);
    const metrics = ['roc_auc_mean', 'pr_auc_mean', 'f1_mean'];
    const labels = ['ROC-AUC', 'PR-AUC', 'F1-Score'];
    const colors = [COLORS.blue, COLORS.green, COLORS.orange];

    const datasets = models.map((name, i) => ({
        label: name,
        data: metrics.map(m => cv[name][m]),
        borderColor: colors[i],
        backgroundColor: colors[i] + '20',
        pointBackgroundColor: colors[i],
        pointRadius: 4,
        borderWidth: 2,
    }));

    chartInstances['chart-radar'] = new Chart(ctx, {
        type: 'radar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#94A3B8', font: { family: 'Inter', size: 12 } },
                },
            },
            scales: {
                r: {
                    angleLines: { color: 'rgba(51, 65, 85, 0.3)' },
                    grid: { color: 'rgba(51, 65, 85, 0.3)' },
                    pointLabels: { color: '#94A3B8', font: { family: 'Inter', size: 12 } },
                    ticks: { display: false },
                    suggestedMin: 0.2,
                    suggestedMax: 0.95,
                },
            },
        },
    });
}

function renderFeatureImportance(features) {
    destroyChart('chart-feature-importance');
    const ctx = document.getElementById('chart-feature-importance');
    if (!ctx) return;

    const defaults = getChartDefaults();
    defaults.indexAxis = 'y';
    defaults.plugins.legend = { display: false };

    const reversed = [...features].reverse();

    chartInstances['chart-feature-importance'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: reversed.map(f => f.feature),
            datasets: [{
                data: reversed.map(f => f.importance),
                backgroundColor: reversed.map((_, i) => {
                    const ratio = i / reversed.length;
                    const r = Math.round(30 + ratio * 7);
                    const g = Math.round(42 + ratio * 57);
                    const b = Math.round(71 + ratio * 164);
                    return `rgb(${r}, ${g}, ${b})`;
                }),
                borderRadius: 4,
                barThickness: 16,
            }],
        },
        options: defaults,
    });
}

// --- Curseur de seuil interactif ---
function setupThresholdSlider() {
    const slider = document.getElementById('slider-threshold');
    const valueLabel = document.getElementById('threshold-value');
    if (!slider || !valueLabel) return;

    slider.addEventListener('input', debounceThreshold(async () => {
        const threshold = slider.value / 100;
        valueLabel.textContent = threshold.toFixed(2);

        const confusion = await apiFetch(`/api/model/confusion?threshold=${threshold}`);
        if (confusion) renderConfusionMatrix(confusion);
    }, 200));
}

function debounceThreshold(fn, delay) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
    };
}
