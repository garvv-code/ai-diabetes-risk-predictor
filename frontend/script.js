/**
 * AI Diabetes Risk Predictor — Frontend Logic
 * =============================================
 * Handles form interaction, API communication,
 * and result rendering.
 */

// ──────────────────────────────────────────────
// ★  API URL Configuration
// ──────────────────────────────────────────────
// Replace this with your Render backend URL after deploying, e.g.:
// const RENDER_URL = "https://diabetes-risk-api.onrender.com";
const RENDER_URL = "https://YOUR-BACKEND-NAME.onrender.com";

// Auto-detect: use localhost for development, Render URL for production
const API_URL = window.location.hostname === "localhost" ||
                window.location.hostname === "127.0.0.1"
  ? "http://localhost:5000"
  : RENDER_URL;

// ──────────────────────────────────────────────
// DOM references
// ──────────────────────────────────────────────
const form       = document.getElementById("predict-form");
const submitBtn  = document.getElementById("submit-btn");
const resultCard = document.getElementById("result-card");
const resultIcon = document.getElementById("result-icon");
const resultTitle = document.getElementById("result-title");
const resultDesc = document.getElementById("result-desc");
const probPct    = document.getElementById("prob-pct");
const probBar    = document.getElementById("prob-bar");

// Slider fields — keep their <output> labels in sync
const sliders = ["pregnancies", "skin", "age"];

sliders.forEach((id) => {
  const slider = document.getElementById(id);
  const output = document.getElementById(`${id}-val`);
  if (slider && output) {
    slider.addEventListener("input", () => {
      output.textContent = slider.value;
    });
  }
});

// ──────────────────────────────────────────────
// Form submission
// ──────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Collect values
  const payload = {
    pregnancies: Number(document.getElementById("pregnancies").value),
    glucose:     Number(document.getElementById("glucose").value),
    bp:          Number(document.getElementById("bp").value),
    skin:        Number(document.getElementById("skin").value),
    insulin:     Number(document.getElementById("insulin").value),
    bmi:         Number(document.getElementById("bmi").value),
    dpf:         Number(document.getElementById("dpf").value),
    age:         Number(document.getElementById("age").value),
  };

  // Basic client-side validation
  for (const [key, val] of Object.entries(payload)) {
    if (isNaN(val)) {
      showError(`Please enter a valid number for "${key}".`);
      return;
    }
  }

  // UI → loading state
  submitBtn.classList.add("loading");
  submitBtn.disabled = true;
  resultCard.classList.add("hidden");

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      const msg = data.details
        ? data.details.join("\n")
        : data.error || "Something went wrong.";
      showError(msg);
      return;
    }

    renderResult(data);
  } catch (err) {
    showError(
      "Unable to reach the prediction server.\nMake sure the backend is running."
    );
    console.error(err);
  } finally {
    submitBtn.classList.remove("loading");
    submitBtn.disabled = false;
  }
});

// ──────────────────────────────────────────────
// Render prediction result
// ──────────────────────────────────────────────
function renderResult({ prediction, probability }) {
  const isHighRisk = prediction === 1;
  const pct = Math.round(probability * 100);

  // Set card theme
  resultCard.className = `card glass result-card ${
    isHighRisk ? "high-risk" : "low-risk"
  }`;

  // Icon & title
  resultIcon.textContent = isHighRisk ? "⚠️" : "✅";
  resultTitle.textContent = isHighRisk ? "High Risk" : "Low Risk";

  // Description
  resultDesc.textContent = isHighRisk
    ? "The model predicts a higher likelihood of diabetes. Please consult a healthcare professional."
    : "The model predicts a lower likelihood of diabetes. Keep maintaining a healthy lifestyle!";

  // Probability bar — animate after a brief delay
  probPct.textContent = `${pct}%`;
  probBar.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      probBar.style.width = `${pct}%`;
    });
  });

  // Show the card
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ──────────────────────────────────────────────
// Error display (reuses the result card)
// ──────────────────────────────────────────────
function showError(message) {
  resultCard.className = "card glass result-card high-risk";
  resultIcon.textContent = "❌";
  resultTitle.textContent = "Error";
  resultDesc.textContent = message;
  probPct.textContent = "—";
  probBar.style.width = "0%";
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "center" });
}
