/**
 * =============================================================================
 *  exploration.js — Page Exploration
 *  Distributions interactives, corrélations, boxplots
 * =============================================================================
 */

let explorationLoaded = false;

async function loadExploration() {
    if (explorationLoaded) return;

    // Charger la liste des features
    const features = await apiFetch('/api/features');
    if (!features) return;

    const select = document.getElementById('select-feature');
    if (select) {
        select.innerHTML = features.map(f =>
            `<option value="${f}" ${f === 'conso_mean' ? 'selected' : ''}>${f}</option>`
        ).join('');
        select.addEventListener('change', () => loadFeatureDistribution(select.value));
    }

    // Charger la distribution par défaut
    await loadFeatureDistribution('conso_mean');

    // Charger les corrélations
    await loadCorrelations();

    explorationLoaded = true;
}

async function loadFeatureDistribution(feature) {
    const data = await apiFetch(`/api/distribution/${encodeURIComponent(feature)}`);
    if (!data || data.error) return;

    // --- Histogramme ---
    destroyChart('chart-histogram');
    const ctxHist = document.getElementById('chart-histogram');
    if (ctxHist) {
        const defaults = getChartDefaults();
        defaults.plugins.legend = {
            labels: { color: '#94A3B8', font: { family: 'Inter', size: 12 } },
        };
        defaults.scales.x.title = {
            display: true, text: feature,
            color: '#94A3B8', font: { family: 'Inter', size: 12 },
        };
        defaults.scales.y.title = {
            display: true, text: 'Densité',
            color: '#94A3B8', font: { family: 'Inter', size: 12 },
        };

        chartInstances['chart-histogram'] = new Chart(ctxHist, {
            type: 'bar',
            data: {
                labels: data.bins,
                datasets: [
                    {
                        label: 'Légitime',
                        data: data.legitimate,
                        backgroundColor: 'rgba(37, 99, 235, 0.6)',
                        borderColor: COLORS.blue,
                        borderWidth: 1,
                        borderRadius: 3,
                    },
                    {
                        label: 'Fraude',
                        data: data.fraud,
                        backgroundColor: 'rgba(220, 38, 38, 0.6)',
                        borderColor: COLORS.red,
                        borderWidth: 1,
                        borderRadius: 3,
                    },
                ],
            },
            options: defaults,
        });
    }

    // --- Stats box ---
    const statsBox = document.getElementById('feature-stats');
    if (statsBox && data.stats) {
        statsBox.innerHTML = `
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-label">Moyenne (Légitime)</span>
                    <span class="stat-value" style="color: ${COLORS.blue};">${data.stats.leg_mean}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Moyenne (Fraude)</span>
                    <span class="stat-value" style="color: ${COLORS.red};">${data.stats.fraud_mean}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Médiane (Légitime)</span>
                    <span class="stat-value" style="color: ${COLORS.blue};">${data.stats.leg_median}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Médiane (Fraude)</span>
                    <span class="stat-value" style="color: ${COLORS.red};">${data.stats.fraud_median}</span>
                </div>
            </div>
        `;
    }
}

async function loadCorrelations() {
    const data = await apiFetch('/api/correlation');
    if (!data || data.length === 0) return;

    destroyChart('chart-correlation');
    const ctx = document.getElementById('chart-correlation');
    if (!ctx) return;

    const defaults = getChartDefaults();
    defaults.indexAxis = 'y';
    defaults.plugins.legend = { display: false };
    defaults.scales.x.title = {
        display: true, text: 'Corrélation avec la fraude (valeur absolue)',
        color: '#94A3B8', font: { family: 'Inter', size: 12 },
    };

    const colors = data.map(d => {
        if (d.correlation > 0.1) return COLORS.red;
        if (d.correlation > 0.05) return COLORS.orange;
        return COLORS.blue;
    });

    chartInstances['chart-correlation'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.feature),
            datasets: [{
                data: data.map(d => d.correlation),
                backgroundColor: colors,
                borderRadius: 6,
                barThickness: 18,
            }],
        },
        options: defaults,
    });
}
