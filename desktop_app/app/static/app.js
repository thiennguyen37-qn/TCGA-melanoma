const form = document.getElementById("predict-form");
const imageInput = document.getElementById("image");
const preview = document.getElementById("preview");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");
const submitBtn = document.getElementById("submit-btn");

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) {
        preview.classList.remove("visible");
        return;
    }
    preview.src = URL.createObjectURL(file);
    preview.classList.add("visible");
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    resultBox.classList.add("hidden");
    errorBox.classList.add("hidden");
    submitBtn.disabled = true;
    submitBtn.textContent = "Đang phân tích...";

    const formData = new FormData(form);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Có lỗi xảy ra, thử lại sau.");
        }

        resultBox.className = `result ${data.label}`;
        resultBox.innerHTML = `
            <div class="label">${data.label === "melanoma" ? "⚠️ Melanoma" : "✅ Nevus"}</div>
            <div class="confidence">Độ tin cậy: ${(data.confidence * 100).toFixed(1)}% (xác suất melanoma: ${(data.probability_melanoma * 100).toFixed(1)}%)</div>
            <div class="disclaimer-text">${data.disclaimer}</div>
        `;
        resultBox.classList.remove("hidden");
    } catch (err) {
        errorBox.textContent = err.message;
        errorBox.classList.remove("hidden");
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Phân tích";
    }
});
