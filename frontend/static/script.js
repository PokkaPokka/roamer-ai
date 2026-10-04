let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let authToken = localStorage.getItem("travel_token") || null;
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

// =========================
// Auth
// =========================

// fetch() with the login token attached. A 401 means the token is missing or
// expired, so log out and show the login form again.
async function apiFetch(url, options = {}) {
    const headers = { ...(options.headers || {}) };

    if (authToken) {
        headers["Authorization"] = `Bearer ${authToken}`;
    }

    const response = await fetch(url, { ...options, headers });

    if (response.status === 401) {
        const message = "Your session has expired. Please log in again.";
        logout();
        showError(message);
        throw new Error(message);
    }

    return response;
}

async function submitAuth(mode) {
    hideError();

    const email = document.getElementById("authEmail").value.trim();
    const password = document.getElementById("authPassword").value;

    if (!email || !password) {
        showError("Please enter your email and password.");
        return;
    }

    const buttons = [document.getElementById("loginBtn"), document.getElementById("registerBtn")];
    buttons.forEach(button => button.disabled = true);

    try {
        const response = await fetch(`/api/auth/${mode}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Could not log in.");
        }

        authToken = data.token;
        localStorage.setItem("travel_token", authToken);
        localStorage.setItem("travel_email", data.email);

        document.getElementById("authPassword").value = "";
        showApp();

    } catch (error) {
        showError(error.message);
    } finally {
        buttons.forEach(button => button.disabled = false);
    }
}

function logout() {
    authToken = null;
    localStorage.removeItem("travel_token");
    localStorage.removeItem("travel_email");

    // The saved thread belongs to this user, so forget it too.
    newTrip();
    document.getElementById("tripsList").replaceChildren();
    showAuth();
}

function showAuth() {
    document.getElementById("authCard").classList.remove("hidden");
    document.getElementById("plannerCard").classList.add("hidden");
    document.getElementById("tripsSection").classList.add("hidden");
    document.getElementById("accountBar").classList.add("hidden");
}

function showApp() {
    document.getElementById("authCard").classList.add("hidden");
    document.getElementById("plannerCard").classList.remove("hidden");
    document.getElementById("tripsSection").classList.remove("hidden");
    document.getElementById("accountBar").classList.remove("hidden");
    document.getElementById("accountEmail").textContent = localStorage.getItem("travel_email") || "";

    loadTrips();
    restorePlan();
}

// =========================
// My Trips
// =========================

async function loadTrips() {
    try {
        const response = await apiFetch("/api/trips");
        const data = await response.json();

        if (!response.ok || !data.success) {
            return;
        }

        renderTrips(data.trips);
    } catch (error) {
        // apiFetch already handled an expired session; the list is not critical.
    }
}

function renderTrips(trips) {
    const list = document.getElementById("tripsList");

    // Built with textContent rather than innerHTML, so a trip title can't inject HTML.
    const items = trips.map(trip => {
        const button = document.createElement("button");
        button.classList.toggle("active", trip.thread_id === currentThreadId);
        button.onclick = () => openTrip(trip.thread_id);

        const title = document.createElement("span");
        title.textContent = trip.title;

        const date = document.createElement("span");
        date.className = "trip-date";
        date.textContent = new Date(trip.updated_at).toLocaleDateString();

        button.append(title, date);

        const item = document.createElement("li");
        item.append(button);
        return item;
    });

    list.replaceChildren(...items);
    document.getElementById("tripsEmpty").classList.toggle("hidden", trips.length > 0);
}

function openTrip(threadId) {
    hideError();
    currentThreadId = threadId;
    localStorage.setItem("travel_thread_id", threadId);
    restorePlan();
    loadTrips();
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

    document.querySelectorAll("#tripsList button.active").forEach(button => button.classList.remove("active"));
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
        const response = await apiFetch(`/api/travel/${encodeURIComponent(currentThreadId)}`);
        const data = await response.json();

        if (response.ok && data.success) {
            showResult(data.answer, currentThreadId);
            setMode(true);
            return;
        }
    } catch (error) {
        // Fall through and start a new trip.
        if (!authToken) {
            return;
        }
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
        const response = await apiFetch("/api/travel", {
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
        loadTrips();

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
        filename: "roamer-travel-plan.pdf",
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
    if (event.ctrlKey && event.key === "Enter" && authToken) {
        sendMessage();
    }
});

if (authToken) {
    showApp();
} else {
    showAuth();
}
