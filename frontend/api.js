const API_BASE = "http://127.0.0.1:8000";

export async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        return await res.json();
    } catch (e) {
        return { status: "error" };
    }
}

export async function fetchExperiments() {
    try {
        const res = await fetch(`${API_BASE}/experiments/`);
        return await res.json();
    } catch (e) {
        return [];
    }
}

export async function predictClinical(patientData) {
    const res = await fetch(`${API_BASE}/predict/clinical`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patientData)
    });
    return await res.json();
}

export async function extractBrainTwin(patientData) {
    const res = await fetch(`${API_BASE}/brain-twin/extract`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patientData)
    });
    return await res.json();
}

export async function fetchPatientHistory(subjectId) {
    try {
        const res = await fetch(`${API_BASE}/patients/${subjectId}/history`);
        if (!res.ok) throw new Error("Not found");
        return await res.json();
    } catch (e) {
        return { history: [] };
    }
}
