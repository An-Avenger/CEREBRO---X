import { checkHealth, fetchExperiments, predictClinical, extractBrainTwin, fetchPatientHistory } from './api.js';

let twinChart = null;

// Initialize
document.addEventListener("DOMContentLoaded", async () => {
    await updateHealthStatus();
    await loadExperiments();
    initChart();

    document.getElementById("patient-form").addEventListener("submit", handlePredict);
    document.getElementById("btn-load-history").addEventListener("click", handleLoadHistory);
});

async function updateHealthStatus() {
    const statusDiv = document.getElementById("api-status");
    const statusText = statusDiv.querySelector(".text");
    
    const health = await checkHealth();
    
    if (health.status === "ok") {
        statusDiv.className = "status-indicator ok";
        statusText.textContent = "API Online";
    } else {
        statusDiv.className = "status-indicator error";
        statusText.textContent = "API Offline";
    }
}

async function loadExperiments() {
    const exps = await fetchExperiments();
    const tbody = document.getElementById("experiments-body");
    tbody.innerHTML = "";

    exps.forEach(exp => {
        const tr = document.createElement("tr");
        
        let acc = "-", bacc = "-";
        if (exp.metrics) {
            // Find best metric depending on structure
            if (exp.metrics["Logistic Regression"] && exp.metrics["Logistic Regression"].test) {
                acc = (exp.metrics["Logistic Regression"].test.accuracy * 100).toFixed(1) + "%";
                bacc = (exp.metrics["Logistic Regression"].test.balanced_accuracy * 100).toFixed(1) + "%";
            } else if (exp.metrics.models && exp.metrics.models.temporal_gru) {
                acc = (exp.metrics.models.temporal_gru.accuracy * 100).toFixed(1) + "%";
                bacc = (exp.metrics.models.temporal_gru.balanced_accuracy * 100).toFixed(1) + "%";
            } else if (exp.metrics.test_metrics) {
                acc = (exp.metrics.test_metrics.accuracy * 100).toFixed(1) + "%";
                bacc = (exp.metrics.test_metrics.balanced_accuracy * 100).toFixed(1) + "%";
            } else if (exp.metrics.binary_disease_vs_control) {
                acc = (exp.metrics.binary_disease_vs_control.accuracy * 100).toFixed(1) + "%";
                bacc = (exp.metrics.binary_disease_vs_control.balanced_accuracy * 100).toFixed(1) + "%";
            } else if (exp.metrics.ablation && exp.metrics.ablation.C_clinical_plus_mri) {
                acc = (exp.metrics.ablation.C_clinical_plus_mri.accuracy * 100).toFixed(1) + "%";
                bacc = (exp.metrics.ablation.C_clinical_plus_mri.balanced_accuracy * 100).toFixed(1) + "%";
            }
        }

        tr.innerHTML = `
            <td>Phase ${exp.phase}</td>
            <td><strong>${exp.experiment_id}</strong></td>
            <td>${exp.description.split("—")[0]}</td>
            <td>${acc}</td>
            <td>${bacc}</td>
        `;
        tbody.appendChild(tr);
    });
}

function initChart() {
    const ctx = document.getElementById('twinChart').getContext('2d');
    
    // Gradient
    let gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(139, 92, 246, 0.5)'); // purple
    gradient.addColorStop(1, 'rgba(139, 92, 246, 0.0)');

    twinChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 64}, (_, i) => `D${i+1}`),
            datasets: [{
                label: 'Z_t (Brain State)',
                data: Array(64).fill(0),
                borderColor: '#8b5cf6',
                backgroundColor: gradient,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, min: -2, max: 2 },
                x: { grid: { display: false }, ticks: { display: false } }
            },
            plugins: {
                legend: { display: false }
            },
            animation: { duration: 1500, easing: 'easeInOutQuart' }
        }
    });
}

async function handlePredict(e) {
    e.preventDefault();

    const btn = e.target.querySelector("button");
    btn.textContent = "Processing...";
    btn.disabled = true;

    // Single visit payload for simplicity, though the model takes sequences
    // We'll pass a dummy baseline and the current as next
    const currentVisit = {
        age: parseFloat(document.getElementById("age").value),
        educ: parseFloat(document.getElementById("educ").value),
        ses: parseFloat(document.getElementById("ses").value),
        mmse: parseFloat(document.getElementById("mmse").value),
        cdr: parseFloat(document.getElementById("cdr").value),
        nwbv: parseFloat(document.getElementById("nwbv").value),
        etiv: parseFloat(document.getElementById("etiv").value),
        asf: parseFloat(document.getElementById("asf").value)
    };

    const payload = {
        subject_id: document.getElementById("subject_id").value || "UI_TEST",
        visits: [currentVisit]
    };

    try {
        const [predictRes, twinRes] = await Promise.all([
            predictClinical(payload),
            extractBrainTwin(payload)
        ]);

        renderPrediction(predictRes);
        renderTwin(twinRes);

    } catch (err) {
        console.error(err);
        alert("Failed to process prediction. Check API connection.");
    } finally {
        btn.textContent = "Run Prediction & Extract Twin";
        btn.disabled = false;
    }
}

function renderPrediction(res) {
    const container = document.getElementById("prediction-result");
    const probsContainer = document.getElementById("probabilities-container");
    
    container.innerHTML = `<div class="prediction-title">${res.predicted_cdr_label}</div>`;
    probsContainer.innerHTML = "";

    Object.entries(res.class_probabilities).forEach(([cls, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const div = document.createElement("div");
        div.className = "prob-bar-container";
        div.innerHTML = `
            <div class="prob-label">
                <span>${cls}</span>
                <span>${pct}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width: 0%"></div>
            </div>
        `;
        probsContainer.appendChild(div);

        // Animate fill
        setTimeout(() => {
            div.querySelector(".prob-fill").style.width = `${pct}%`;
        }, 50);
    });
}

function renderTwin(res) {
    if (!res.trajectories || res.trajectories.length === 0) return;
    
    // Get the last step of the trajectory
    const z_t = res.trajectories[res.trajectories.length - 1];
    
    twinChart.data.datasets[0].data = z_t;
    twinChart.update();
}

async function handleLoadHistory() {
    const subjectId = document.getElementById("subject_id").value;
    const container = document.getElementById("history-container");
    
    if (!subjectId) {
        container.innerHTML = `<div class="result-placeholder">Please enter a Subject ID</div>`;
        return;
    }
    
    container.innerHTML = `<div class="result-placeholder">Loading...</div>`;
    
    const data = await fetchPatientHistory(subjectId);
    
    if (!data.history || data.history.length === 0) {
        container.innerHTML = `<div class="result-placeholder">No history found for ${subjectId}</div>`;
        return;
    }
    
    let html = '<ul style="list-style-type: none; padding: 0;">';
    data.history.forEach(r => {
        const date = new Date(r.timestamp).toLocaleString();
        html += `
            <li style="border-bottom: 1px solid rgba(255,255,255,0.1); padding: 8px 0; margin-bottom: 8px;">
                <div style="font-size: 0.8rem; color: #94a3b8;">${date}</div>
                <div>Age: ${r.age} | MMSE: ${r.mmse}</div>
                <div style="color: var(--accent-blue); font-weight: 600;">Predicted: ${r.predicted_cdr_label}</div>
            </li>
        `;
    });
    html += '</ul>';
    
    container.innerHTML = html;
}
