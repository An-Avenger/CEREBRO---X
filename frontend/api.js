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

/**
 * Real SHAP GradientExplainer attributions for the Clinical GRU.
 * @param {Object} predictionRequest - { subject_id, visits: [...] }
 */
export async function fetchClinicalExplain(predictionRequest) {
    const res = await fetch(`${API_BASE}/explain/clinical`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(predictionRequest)
    });
    return await res.json();
}

/**
 * Real 3D Grad-CAM for an uploaded NIfTI MRI file.
 * @param {File} niftiFile - .nii or .nii.gz file
 * @param {number|null} targetClass - optional CDR class to explain (0-3)
 */
export async function fetchMRIGradCAM(niftiFile, targetClass = null) {
    const form = new FormData();
    form.append("file", niftiFile);
    const url = targetClass !== null
        ? `${API_BASE}/explain/mri?target_class=${targetClass}`
        : `${API_BASE}/explain/mri`;
    const res = await fetch(url, { method: "POST", body: form });
    return await res.json();
}
