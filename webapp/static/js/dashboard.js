/**
 * =============================================================================
 *  dashboard.js — Page Dashboard
 *  KPI globaux, fraude par région, timeline
 * =============================================================================
 */

let dashboardLoaded = false;

async function loadDashboard() {
    if (dashboardLoaded) return;

    const [stats, regions, timeline] = await Promise.all([
        apiFetch('/api/stats'),
        apiFetch('/api/fraud-by-region'),
        apiFetch('/api/timeline'),
    ]);

    if (!stats) return;

    // --- KPI Cards ---
    animateCounter(document.querySelector('#kpi-clients .kpi-value'), stats.n_clients);
    animateCounter(document.querySelector('#kpi-invoices .kpi-value'), stats.n_invoices);
    animateCounter(document.querySelector('#kpi-fraudsters .kpi-value'), stats.n_fraud);
    animateCounter(
        document.querySelector('#kpi-fraud-rate .kpi-value'),
        Math.round(stats.fraud_rate * 100) / 100,
        1200,
        '%'
    );

    // Mettre à jour le texte du taux de fraude avec précision
    setTimeout(() => {
        document.querySelector('#kpi-fraud-rate .kpi-value').textContent = stats.fraud_rate + '%';
    }, 1300);

    // --- Fraude par Région (barres horizontales) ---
    if (regions && regions.length > 0) {
        destroyChart('chart-fraud-region');
        const ctx = document.getElementById('chart-fraud-region');
        if (ctx) {
            const labels = regions.map(r => `Région ${r.region}`);
            const data = regions.map(r => r.fraud_rate);
            const colors = data.map(v => {
                if (v > 10) return COLORS.red;
                if (v > 6) return COLORS.orange;
                return COLORS.blue;
            });

            const defaults = getChartDefaults();
            defaults.indexAxis = 'y';
            defaults.plugins.legend = { display: false };
            defaults.scales.x.title = {
                display: true, text: 'Taux de fraude (%)',
                color: '#94A3B8', font: { family: 'Inter', size: 12 },
            };

            chartInstances['chart-fraud-region'] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: colors,
                        borderRadius: 6,
                        borderSkipped: false,
                        barThickness: 20,
                    }],
                },
                options: defaults,
            });
        }
    }

    // --- Evolution Temporelle ---
    if (timeline && timeline.length > 0) {
        destroyChart('chart-timeline');
        const ctx = document.getElementById('chart-timeline');
        if (ctx) {
            const defaults = getChartDefaults();
            defaults.plugins.legend = {
                labels: { color: '#94A3B8', font: { family: 'Inter', size: 12 } },
            };
            defaults.scales.y.title = {
                display: true, text: 'Nombre de clients',
                color: '#94A3B8', font: { family: 'Inter', size: 12 },
            };
            defaults.scales.y1 = {
                position: 'right',
                title: {
                    display: true, text: 'Taux de fraude (%)',
                    color: '#94A3B8', font: { family: 'Inter', size: 12 },
                },
                ticks: { color: '#94A3B8', font: { family: 'Inter', size: 11 } },
                grid: { display: false },
            };

            chartInstances['chart-timeline'] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: timeline.map(t => t.year),
                    datasets: [
                        {
                            label: 'Clients',
                            data: timeline.map(t => t.total),
                            backgroundColor: 'rgba(37, 99, 235, 0.5)',
                            borderColor: COLORS.blue,
                            borderWidth: 1,
                            borderRadius: 4,
                            yAxisID: 'y',
                            order: 2,
                        },
                        {
                            label: 'Taux de fraude (%)',
                            data: timeline.map(t => t.fraud_rate),
                            borderColor: COLORS.red,
                            backgroundColor: 'rgba(220, 38, 38, 0.1)',
                            type: 'line',
                            tension: 0.4,
                            pointRadius: 4,
                            pointBackgroundColor: COLORS.red,
                            borderWidth: 2.5,
                            fill: true,
                            yAxisID: 'y1',
                            order: 1,
                        },
                    ],
                },
                options: defaults,
            });
        }
    }

    // --- Métriques du modèle (cards additionnelles) ---
    const modelCards = document.getElementById('model-kpi-cards');
    if (modelCards) {
        modelCards.innerHTML = `
            <div class="kpi-card kpi-accent-green">
                <div class="kpi-value">${stats.roc_auc}</div>
                <div class="kpi-label">ROC-AUC</div>
            </div>
            <div class="kpi-card kpi-accent-purple">
                <div class="kpi-value">${stats.pr_auc}</div>
                <div class="kpi-label">PR-AUC</div>
            </div>
            <div class="kpi-card kpi-accent-orange">
                <div class="kpi-value">${stats.model_name}</div>
                <div class="kpi-label">Modèle Sélectionné</div>
            </div>
        `;
    }

    dashboardLoaded = true;
}
