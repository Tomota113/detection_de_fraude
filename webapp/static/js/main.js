/**
 * =============================================================================
 *  main.js — Navigation SPA & Utilitaires
 *  Détection de Fraude Énergétique — STEG Tunisie
 * =============================================================================
 */

// =============================================================================
// NAVIGATION SPA
// =============================================================================

const sections = ['dashboard', 'scoring', 'exploration', 'performance', 'suspects'];

function navigateTo(page) {
    // Masquer toutes les sections
    sections.forEach(s => {
        const section = document.getElementById(`section-${s}`);
        const nav = document.getElementById(`nav-${s}`);
        if (section) section.classList.remove('active');
        if (nav) nav.classList.remove('active');
    });

    // Afficher la section demandée
    const activeSection = document.getElementById(`section-${page}`);
    const activeNav = document.getElementById(`nav-${page}`);
    if (activeSection) {
        activeSection.classList.add('active');
        activeSection.classList.add('fade-in');
    }
    if (activeNav) activeNav.classList.add('active');

    // Charger les données de la page si nécessaire
    switch (page) {
        case 'dashboard':
            if (typeof loadDashboard === 'function') loadDashboard();
            break;
        case 'scoring':
            // Chargé au clic sur le bouton
            break;
        case 'exploration':
            if (typeof loadExploration === 'function') loadExploration();
            break;
        case 'performance':
            if (typeof loadPerformance === 'function') loadPerformance();
            break;
        case 'suspects':
            if (typeof loadSuspects === 'function') loadSuspects();
            break;
    }
}

// =============================================================================
// UTILITAIRES
// =============================================================================

/**
 * Appel API simplifié.
 */
async function apiFetch(endpoint) {
    try {
        const response = await fetch(endpoint);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error(`Erreur API ${endpoint}:`, error);
        return null;
    }
}

/**
 * Animation de comptage d'un nombre.
 */
function animateCounter(element, target, duration = 1200, suffix = '') {
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Easing out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (target - start) * eased);
        element.textContent = formatNumber(current) + suffix;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

/**
 * Formate un nombre avec séparateurs de milliers.
 */
function formatNumber(n) {
    return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

/**
 * Crée les options par défaut pour les graphiques Chart.js (dark mode).
 */
function getChartDefaults() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: '#94A3B8',
                    font: { family: 'Inter', size: 12 },
                },
            },
            tooltip: {
                backgroundColor: '#1E293B',
                titleColor: '#FFFFFF',
                bodyColor: '#94A3B8',
                borderColor: '#334155',
                borderWidth: 1,
                cornerRadius: 8,
                padding: 12,
                titleFont: { family: 'Inter', weight: '600' },
                bodyFont: { family: 'Inter' },
            },
        },
        scales: {
            x: {
                ticks: { color: '#94A3B8', font: { family: 'Inter', size: 11 } },
                grid: { color: 'rgba(51, 65, 85, 0.3)' },
            },
            y: {
                ticks: { color: '#94A3B8', font: { family: 'Inter', size: 11 } },
                grid: { color: 'rgba(51, 65, 85, 0.3)' },
            },
        },
    };
}

/**
 * Détruit un graphique Chart.js s'il existe.
 */
const chartInstances = {};
function destroyChart(id) {
    if (chartInstances[id]) {
        chartInstances[id].destroy();
        delete chartInstances[id];
    }
}

/**
 * Couleurs de la palette.
 */
const COLORS = {
    blue: '#2563EB',
    blueLght: '#93C5FD',
    red: '#DC2626',
    redLight: '#FCA5A5',
    green: '#059669',
    greenLight: '#6EE7B7',
    orange: '#D97706',
    purple: '#7C3AED',
    gray: '#6B7280',
    text: '#FFFFFF',
    textSec: '#94A3B8',
};

// =============================================================================
// INITIALISATION
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Attacher les handlers de navigation
    sections.forEach(s => {
        const nav = document.getElementById(`nav-${s}`);
        if (nav) {
            nav.addEventListener('click', (e) => {
                e.preventDefault();
                navigateTo(s);
            });
        }
    });

    // Charger le dashboard par défaut
    navigateTo('dashboard');
});
