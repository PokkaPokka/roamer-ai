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

  const buttons = [
    document.getElementById("loginBtn"),
    document.getElementById("registerBtn"),
  ];
  buttons.forEach((button) => (button.disabled = true));

  try {
    const response = await fetch(`/api/auth/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
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
    buttons.forEach((button) => (button.disabled = false));
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
  document.getElementById("accountEmail").textContent =
    localStorage.getItem("travel_email") || "";

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
  const items = trips.map((trip) => {
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
  document
    .getElementById("tripsEmpty")
    .classList.toggle("hidden", trips.length > 0);
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
  placeholder:
    "Plan a 3 day trip from Melbourne to Tokyo including flights, hotels and sightseeing...",
  button: "Generate Plan",
};

const FEEDBACK_TEXT = {
  title: "Want to change anything?",
  hint: 'Tell the planner what to change, e.g. "make it 5 days" or "add more food spots on day 2".',
  placeholder: "Your feedback on the plan...",
  button: "Revise Plan",
};

const ROUTE_LABELS = {
  plan: "New plan",
  new_search: "Updated with new flight and hotel searches",
  revise: "Revised from your feedback",
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
  stopPlanning();
  document.getElementById("progressPanel").classList.add("hidden");
  hideFlightChoice();
  currentThreadId = null;
  latestAnswerMarkdown = "";
  localStorage.removeItem("travel_thread_id");

  document
    .querySelectorAll("#tripsList button.active")
    .forEach((button) => button.classList.remove("active"));
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
    const response = await apiFetch(
      `/api/travel/${encodeURIComponent(currentThreadId)}`,
    );
    const data = await response.json();

    if (response.ok && data.success) {
      // A trip paused on a flight choice has options but no plan yet.
      if (data.flight_options) {
        showFlightChoice(data.flight_options);
      } else {
        hideFlightChoice();
      }

      if (data.answer) {
        showResult(data.answer, currentThreadId);
      } else {
        document.getElementById("resultSection").classList.add("hidden");
      }
      setMode(Boolean(data.answer));
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

// Markdown -> HTML, cleaned by DOMPurify so model output can't add scripts.
// Without both libraries, show plain text instead.
function renderMarkdown(element, markdown) {
  if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
    element.innerHTML = DOMPurify.sanitize(marked.parse(markdown));
  } else {
    element.innerText = markdown;
  }
}

function showResult(answer, threadId, route, scroll = true) {
  latestAnswerMarkdown = answer;

  const resultSection = document.getElementById("resultSection");
  const threadInfo = document.getElementById("threadInfo");

  renderMarkdown(document.getElementById("resultBox"), answer);

  const routeLabel = ROUTE_LABELS[route] ? `${ROUTE_LABELS[route]} · ` : "";
  threadInfo.textContent = `${routeLabel}Thread ID: ${threadId}`;

  resultSection.classList.remove("hidden");

  if (scroll) {
    resultSection.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }
}

// =========================
// Live progress
// =========================

// Heartbeats arrive every 10 s, so 30 s with no data at all means the connection
// is gone. The model can read its prompt for 2 minutes before writing anything,
// so only warn about slowness after 4 minutes without any progress event.
const CONNECTION_LOST_MS = 30000;
const SLOW_PROGRESS_MS = 240000;
const RENDER_EVERY_MS = 200;

const STEP_ICONS = {
  waiting: "○",
  running: "●",
  done: "✓",
  failed: "✗",
  skipped: "–",
  paused: "⏸",
};

let planController = null; // AbortController of the running request
let progressTimer = null; // checks once a second whether the run looks stuck
let lastDataAt = 0; // any data, including heartbeats
let lastProgressAt = 0; // a real event: step, status or token
let steps = new Map(); // node -> { element, status }
let liveRenderTimer = null;

function createStep(label) {
  const element = document.createElement("li");
  element.className = "step waiting";

  const icon = document.createElement("span");
  icon.className = "step-icon";
  icon.textContent = STEP_ICONS.waiting;

  const name = document.createElement("span");
  name.className = "step-label";
  name.textContent = label;

  const message = document.createElement("span");
  message.className = "step-message";

  element.append(icon, name, message);
  return { element, status: "waiting" };
}

// The server sends the step list once at the start and again when the route is
// known. Steps already shown keep their state.
function setSteps(list) {
  const items = list.map(({ node, label }) => {
    if (!steps.has(node)) {
      steps.set(node, createStep(label));
    }
    return steps.get(node).element;
  });

  document.getElementById("stepList").replaceChildren(...items);
}

function setStepStatus(node, status) {
  const step = steps.get(node);
  if (!step) {
    return;
  }

  step.status = status;
  step.element.className = `step ${status}`;
  step.element.querySelector(".step-icon").textContent = STEP_ICONS[status];
}

function setStepMessage(node, message) {
  const step = steps.get(node);
  if (step) {
    step.element.querySelector(".step-message").textContent = message;
  }
}

function showProgressWarning(text) {
  const warning = document.getElementById("progressWarning");
  warning.textContent = text;
  warning.classList.toggle("hidden", !text);
}

function tickProgress() {
  const now = Date.now();
  const anyRunning = [...steps.values()].some(
    (step) => step.status === "running",
  );

  if (now - lastDataAt > CONNECTION_LOST_MS) {
    showProgressWarning(
      "No response from the server. The connection may have dropped; press Stop and try again.",
    );
  } else if (anyRunning && now - lastProgressAt > SLOW_PROGRESS_MS) {
    showProgressWarning(
      "This is taking longer than usual. You can keep waiting or press Stop.",
    );
  } else {
    showProgressWarning("");
  }
}

function startProgress() {
  steps = new Map();
  document.getElementById("stepList").replaceChildren();
  document.getElementById("progressTitle").textContent = "Planning your trip";
  document.getElementById("stopBtn").classList.remove("hidden");
  document.getElementById("progressPanel").classList.remove("hidden");
  showProgressWarning("");

  lastDataAt = lastProgressAt = Date.now();
  clearInterval(progressTimer);
  progressTimer = setInterval(tickProgress, 1000);
  tickProgress();
}

function endProgress(title) {
  clearInterval(progressTimer);
  progressTimer = null;
  clearTimeout(liveRenderTimer);
  liveRenderTimer = null;

  // Anything still running didn't finish.
  steps.forEach((step, node) => {
    if (step.status === "running") {
      setStepStatus(node, "failed");
    }
  });

  document.getElementById("progressTitle").textContent = title;
  document.getElementById("stopBtn").classList.add("hidden");
  showProgressWarning("");
}

function stopPlanning() {
  if (planController) {
    planController.abort();
  }
}

// Shows the plan while the model writes it, re-rendered at most every 200 ms.
function showLivePlan(markdown) {
  if (liveRenderTimer) {
    return;
  }

  liveRenderTimer = setTimeout(() => {
    liveRenderTimer = null;

    const resultSection = document.getElementById("resultSection");
    const firstRender = resultSection.classList.contains("hidden");

    renderMarkdown(document.getElementById("resultBox"), markdown());
    document.getElementById("threadInfo").textContent = "Writing...";
    resultSection.classList.remove("hidden");

    if (firstRender) {
      resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, RENDER_EVERY_MS);
}

// Reads Server-Sent Events from a fetch() response. EventSource can't be used
// because it can't send the Authorization header.
async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        return;
      }

      lastDataAt = Date.now();
      buffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        // Lines starting with ":" are heartbeats and carry no data.
        const data = block
          .split("\n")
          .filter((line) => line.startsWith("data: "))
          .map((line) => line.slice(6))
          .join("\n");

        if (data) {
          onEvent(JSON.parse(data));
        }
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}

// =========================
// Flight choice (the run pauses until the user picks one)
// =========================

function formatPrice(option) {
  return option.price === null
    ? "Price unavailable"
    : `${option.currency} ${option.price.toLocaleString()}`;
}

// "2026-11-20 06:15" -> "06:15"
function timeOf(dateTime) {
  return (dateTime || "").split(" ").pop() || "?";
}

// Days between the departure and arrival dates, shown as "+1 day" for overnight flights.
function daysLater(departure, arrival) {
  const days = Math.round(
    (Date.parse((arrival || "").split(" ")[0]) -
      Date.parse((departure || "").split(" ")[0])) /
      86400000,
  );
  return days > 0 ? ` (+${days} day${days > 1 ? "s" : ""})` : "";
}

// Built with textContent, so nothing from the search can inject HTML.
function flightCard(option) {
  const card = document.createElement("li");
  card.className = "flight-card";

  const price = document.createElement("div");
  price.className = "flight-price";
  price.textContent = formatPrice(option);

  const priceNote = document.createElement("span");
  priceNote.className = "flight-note";
  priceNote.textContent = option.round_trip ? "round trip" : "one way";
  price.append(" ", priceNote);

  const airlines = document.createElement("div");
  airlines.className = "flight-airlines";
  airlines.textContent = option.airlines.join(", ");

  const route = document.createElement("div");
  route.textContent =
    `${option.departure_airport} ${timeOf(option.departure_time)} → ` +
    `${option.arrival_airport} ${timeOf(option.arrival_time)}` +
    daysLater(option.departure_time, option.arrival_time);

  const details = document.createElement("div");
  details.className = "flight-note";
  const stops =
    option.stops === 0
      ? "Direct"
      : `${option.stops} stop${option.stops > 1 ? "s" : ""}`;
  details.textContent = `${option.duration} · ${stops}`;

  const button = document.createElement("button");
  button.className = "auth-primary";
  button.textContent = "Use this flight";
  button.onclick = () => chooseFlight(option.number);

  card.append(price, airlines, route, details, button);
  return card;
}

function showFlightChoice(options) {
  document.getElementById("flightOptions").replaceChildren(...options.map(flightCard));
  const section = document.getElementById("flightChoice");
  section.classList.remove("hidden");
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function hideFlightChoice() {
  document.getElementById("flightChoice").classList.add("hidden");
  document.getElementById("flightOptions").replaceChildren();
}

// 0 means "continue without a flight".
function chooseFlight(number) {
  if (!currentThreadId) {
    return;
  }
  runPlan({ thread_id: currentThreadId, flight_choice: number });
}

// =========================
// Sending a request
// =========================

async function sendMessage() {
  hideError();

  const input = document.getElementById("userInput");
  const message = input.value.trim();

  if (!message) {
    showError("Please enter your travel request first.");
    return;
  }

  await runPlan({ message: message, thread_id: currentThreadId }, () => {
    input.value = "";
  });
}

// Streams one run: a new request, feedback, or the answer to a flight choice.
// onAccepted runs once the server has taken the request (plan done or paused).
async function runPlan(body, onAccepted = () => {}) {
  hideError();
  hideFlightChoice();
  setLoading(true);
  startProgress();
  planController = new AbortController();

  let liveText = "";
  let finished = false;

  try {
    const response = await apiFetch("/api/travel/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: planController.signal,
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || "Something went wrong.");
    }

    await readEvents(response, (event) => {
      lastProgressAt = Date.now();

      switch (event.type) {
        case "steps":
          setSteps(event.steps);
          break;
        case "step":
          setStepStatus(event.node, event.status);
          break;
        case "status":
          setStepMessage(event.node, event.message);
          break;
        case "token":
          // The model has finished reading, so drop "Reading the search results".
          if (!liveText) {
            steps.forEach((step, node) => {
              if (step.status === "running") {
                setStepMessage(node, "");
              }
            });
          }
          liveText += event.text;
          showLivePlan(() => liveText);
          break;
        case "choose":
          // The run is paused and saved; it continues when a flight is picked.
          finished = true;
          endProgress("Pick a flight to continue");

          currentThreadId = event.thread_id;
          localStorage.setItem("travel_thread_id", currentThreadId);

          showFlightChoice(event.options);
          onAccepted();
          loadTrips();
          break;
        case "done":
          finished = true;
          endProgress("Plan ready");

          currentThreadId = event.thread_id;
          localStorage.setItem("travel_thread_id", currentThreadId);

          showResult(event.answer, event.thread_id, event.route, false);
          onAccepted();
          setMode(true);
          loadTrips();
          break;
        case "error":
          throw new Error(event.message);
      }
    });

    if (!finished) {
      throw new Error(
        "The connection closed before the plan was finished. Please try again.",
      );
    }
  } catch (error) {
    const stopped = error.name === "AbortError";
    endProgress(stopped ? "Stopped" : "Planning failed");

    if (!stopped) {
      showError(error.message);
    }

    // Drop the half-written plan and show the saved state (a plan, or the
    // flight cards of a paused trip), if there is one.
    if (currentThreadId) {
      restorePlan();
    } else if (liveText) {
      document.getElementById("resultSection").classList.add("hidden");
    }
  } finally {
    planController = null;
    setLoading(false);
  }
}

function copyResult() {
  const resultBox = document.getElementById("resultBox");
  const text = resultBox.innerText;

  if (!text) {
    return;
  }

  navigator.clipboard
    .writeText(text)
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
      quality: 0.98,
    },
    html2canvas: {
      scale: 2,
      useCORS: true,
      backgroundColor: "#ffffff",
    },
    jsPDF: {
      unit: "in",
      format: "a4",
      orientation: "portrait",
    },
    pagebreak: {
      mode: ["avoid-all", "css", "legacy"],
    },
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

document.addEventListener("keydown", function (event) {
  if (event.ctrlKey && event.key === "Enter" && authToken) {
    sendMessage();
  }
});

if (authToken) {
  showApp();
} else {
  showAuth();
}
