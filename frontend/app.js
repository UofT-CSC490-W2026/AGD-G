const config = window.AGDG_CONFIG;

const form = document.querySelector("#attack-form");
const imageInput = document.querySelector("#chart-image");
const promptInput = document.querySelector("#prompt");
const targetResponseInput = document.querySelector("#target-response");
const submitButton = document.querySelector("#submit-button");
const clearButton = document.querySelector("#clear-button");
const apiStatus = document.querySelector("#api-status");
const apiModel = document.querySelector("#api-model");
const requestMeta = document.querySelector("#request-meta");
const adversarialAnswer = document.querySelector("#adversarial-answer");
const adversarialImage = document.querySelector("#adversarial-image");

function apiUrl(path) {
  return `${config.apiBaseUrl}${path}`;
}

function setStatus(label, isHealthy) {
  apiStatus.textContent = label;
  apiStatus.style.color = isHealthy ? "var(--accent)" : "var(--danger)";
}

function resetOutputs() {
  adversarialAnswer.textContent = "No result yet.";
  adversarialImage.removeAttribute("src");
  requestMeta.textContent = "Waiting for input";
}

async function checkHealth() {
  try {
    const response = await fetch(apiUrl(config.healthPath));
    if (!response.ok) {
      throw new Error(`Health check failed with ${response.status}`);
    }
    const payload = await response.json();
    setStatus("API reachable", true);
    apiModel.textContent = `Mode: ${payload.mode} | Surrogate: ${payload.surrogate}`;
  } catch (error) {
    setStatus("API unavailable", false);
    apiModel.textContent = "Mode: unknown";
    requestMeta.textContent = error.message;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];
  if (!file) {
    requestMeta.textContent = "Choose a chart image first.";
    return;
  }

  submitButton.disabled = true;
  requestMeta.textContent = "Running attack...";
  adversarialAnswer.textContent = "Generating adversarial answer...";

  const formData = new FormData();
  formData.append("image", file);
  formData.append("prompt", promptInput.value.trim());
  formData.append("target_response", targetResponseInput.value.trim());

  try {
    const response = await fetch(apiUrl(config.attackPath), {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Request failed with ${response.status}`);
    }

    adversarialAnswer.textContent = data.adversarial_answer;
    adversarialImage.src = data.adversarial_image;
    requestMeta.textContent = `Mode: ${data.mode} | Evaluator: ${data.evaluation_model}`;
  } catch (error) {
    adversarialAnswer.textContent = error.message;
    requestMeta.textContent = "Attack failed";
  } finally {
    submitButton.disabled = false;
  }
});

clearButton.addEventListener("click", () => {
  form.reset();
  resetOutputs();
});

resetOutputs();
checkHealth();
