let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";

function setPrompt(text) {
    document.getElementById("userInput").value = text;
}

function setLoading(isLoading) {
    const sendBtn = document.getElementById("sendBtn");
    const btnText = document.getElementById("btnText");
    const btnLoader = document.getElementById("btnLoader");

    sendBtn.disabled = isLoading;

    if (isLoading) {
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
    } else {
        btnText.classList.remove("hidden");
        btnLoader.classList.add("hidden");
    }
}

function showError(message) {
    const errorBox = document.getElementById("errorBox");

    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function hideError() {
    const errorBox = document.getElementById("errorBox");

    errorBox.classList.add("hidden");
    errorBox.textContent = "";
}

const NEW_TRIP_TEXT = {
    title: "Where do you want to go?",
    hint: "Example: Plan a 3 day trip from Melbourne to Tokyo leaving 20 November.",
    placeholder: "Plan a 3 day trip from Melbourne to Tokyo including flights, hotels and sightseeing...",
    button: "Generate Plan"
};

const FEEDBACK_TEXT = {
    title: "Want to change anything?",
    hint: "Tell the planner what to change, e.g. \"make it 5 days\" or \"add more food spots on day 2\".",
    placeholder: "Your feedback on the plan...",
    button: "Revise Plan"
};

const ROUTE_LABELS = {
    plan: "New plan",
    new_search: "Updated with new flight and hotel searches",
    revise: "Revised from your feedback"
};

function setMode(hasPlan) {
    const text = hasPlan ? FEEDBACK_TEXT : NEW_TRIP_TEXT;

    document.getElementById("inputTitle").textContent = text.title;
    document.getElementById("inputHint").textContent = text.hint;
    document.getElementById("userInput").placeholder = text.placeholder;
    document.getElementById("btnText").textContent = text.button;
    document.getElementById("quickPrompts").classList.toggle("hidden", hasPlan);
}

function newTrip() {
    currentThreadId = null;
    latestAnswerMarkdown = "";
    localStorage.removeItem("travel_thread_id");

    document.getElementById("resultSection").classList.add("hidden");
    document.getElementById("userInput").value = "";
    hideError();
    setMode(false);
}

async function restorePlan() {
    if (!currentThreadId) {
        setMode(false);
        return;
    }

    try {
        const response = await fetch(`/api/travel/${encodeURIComponent(currentThreadId)}`);
        const data = await response.json();

        if (response.ok && data.success) {
            showResult(data.answer, currentThreadId);
            setMode(true);
            return;
        }
    } catch (error) {
        // Fall through and start a new trip.
    }

    newTrip();
}

function showResult(answer, threadId, route) {
    latestAnswerMarkdown = answer;

    const resultSection = document.getElementById("resultSection");
    const resultBox = document.getElementById("resultBox");
    const threadInfo = document.getElementById("threadInfo");

    if (typeof marked !== "undefined") {
        resultBox.innerHTML = marked.parse(answer);
    } else {
        resultBox.innerText = answer;
    }

    const routeLabel = ROUTE_LABELS[route] ? `${ROUTE_LABELS[route]} · ` : "";
    threadInfo.textContent = `${routeLabel}Thread ID: ${threadId}`;

    resultSection.classList.remove("hidden");

    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

async function sendMessage() {
    hideError();

    const input = document.getElementById("userInput");
    const message = input.value.trim();

    if (!message) {
        showError("Please enter your travel request first.");
        return;
    }

    setLoading(true);

    try {
        const response = await fetch("/api/travel", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                thread_id: currentThreadId
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Something went wrong.");
        }

        currentThreadId = data.thread_id;
        localStorage.setItem("travel_thread_id", currentThreadId);

        showResult(data.answer, data.thread_id, data.route);
        input.value = "";
        setMode(true);

    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

function copyResult() {
    const resultBox = document.getElementById("resultBox");
    const text = resultBox.innerText;

    if (!text) {
        return;
    }

    navigator.clipboard.writeText(text)
        .then(() => {
            const copyBtn = document.querySelector(".copy-btn");
            const oldText = copyBtn.textContent;

            copyBtn.textContent = "Copied!";

            setTimeout(() => {
                copyBtn.textContent = oldText;
            }, 1400);
        })
        .catch(() => {
            showError("Could not copy result.");
        });
}

function downloadPDF() {
    const pdfContent = document.getElementById("pdfContent");

    if (!latestAnswerMarkdown || !pdfContent) {
        showError("No travel plan available to download.");
        return;
    }

    const downloadBtn = document.querySelector(".download-btn");
    const oldText = downloadBtn.textContent;

    downloadBtn.textContent = "Preparing PDF...";
    downloadBtn.disabled = true;

    const options = {
        margin: 0.5,
        filename: "ai-travel-plan.pdf",
        image: {
            type: "jpeg",
            quality: 0.98
        },
        html2canvas: {
            scale: 2,
            useCORS: true,
            backgroundColor: "#ffffff"
        },
        jsPDF: {
            unit: "in",
            format: "a4",
            orientation: "portrait"
        },
        pagebreak: {
            mode: ["avoid-all", "css", "legacy"]
        }
    };

    html2pdf()
        .set(options)
        .from(pdfContent)
        .save()
        .then(() => {
            downloadBtn.textContent = oldText;
            downloadBtn.disabled = false;
        })
        .catch(() => {
            downloadBtn.textContent = oldText;
            downloadBtn.disabled = false;
            showError("Could not download PDF.");
        });
}

document.addEventListener("keydown", function(event) {
    if (event.ctrlKey && event.key === "Enter") {
        sendMessage();
    }
});
restorePlan();
