// The trip on screen (null = a new trip). Separate from the plan being generated
// (activeRun below), which keeps going while you look at another trip.
let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let authToken = localStorage.getItem("travel_token") || null;
let latestAnswerMarkdown = "";
let savedTrips = [];

// The one plan being generated, if any. The server allows one per account.
//   threadId  - set by the server's "start" event for a new trip
//   isNew     - a new trip, not in My Trips until it finishes or pauses
//   title     - shown in My Trips while it runs
//   onScreen  - whether its progress and text are being shown
//   liveText  - the plan written so far
let activeRun = null;

const RUN_IN_PROGRESS =
  "A plan is already being generated. Wait for it to finish or press Stop.";

function setLoading(isLoading) {
  document.getElementById("sendBtn").disabled = isLoading;
  document.getElementById("btnText").classList.toggle("hidden", isLoading);
  document.getElementById("btnLoader").classList.toggle("hidden", !isLoading);
}

// The send button spins only while the running trip is on screen; on any other
// trip it stays clickable and explains that a plan is already running.
function refreshControls() {
  setLoading(Boolean(activeRun && activeRun.onScreen));
}

function errorElement() {
  const appVisible = !document.getElementById("appView").classList.contains("hidden");
  return document.getElementById(appVisible ? "errorBox" : "authError");
}

function showError(message) {
  const errorBox = errorElement();
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function hideError() {
  ["errorBox", "authError"].forEach((id) => {
    const box = document.getElementById(id);
    box.classList.add("hidden");
    box.textContent = "";
  });
}

// =========================
// Toast (news about a plan running in the background)
// =========================

let toastTimer = null;

function showToast(text, { actionLabel = "", onAction = null, isError = false } = {}) {
  const toast = document.getElementById("toast");
  const action = document.getElementById("toastAction");

  document.getElementById("toastText").textContent = text;
  toast.classList.toggle("toast-error", isError);

  action.classList.toggle("hidden", !actionLabel);
  action.textContent = actionLabel;
  action.onclick = () => {
    hideToast();
    if (onAction) {
      onAction();
    }
  };

  toast.classList.remove("hidden");

  // A toast with a button stays until it's used or dismissed.
  clearTimeout(toastTimer);
  if (!actionLabel) {
    toastTimer = setTimeout(hideToast, 5000);
  }
}

function hideToast() {
  clearTimeout(toastTimer);
  document.getElementById("toast").classList.add("hidden");
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
  stopPlanning();
  authToken = null;
  localStorage.removeItem("travel_token");
  localStorage.removeItem("travel_email");

  // The saved thread belongs to this user, so forget it too.
  newTrip();
  savedTrips = [];
  renderTrips();
  showAuth();
}

function showAuth() {
  document.getElementById("authCard").classList.remove("hidden");
  document.getElementById("appView").classList.add("hidden");
  document.getElementById("accountBar").classList.add("hidden");
}

function showApp() {
  hideError();
  document.getElementById("authCard").classList.add("hidden");
  document.getElementById("appView").classList.remove("hidden");
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

    savedTrips = data.trips;
    renderTrips();
  } catch (error) {
    // apiFetch already handled an expired session; the list is not critical.
  }
}

function renderTrips() {
  const items = [];

  // A new trip being planned isn't saved yet; list it so it can be reopened.
  if (activeRun && activeRun.isNew && !savedTrips.some((t) => t.thread_id === activeRun.threadId)) {
    items.push(tripItem({ thread_id: activeRun.threadId, title: activeRun.title }, true));
  }
  savedTrips.forEach((trip) => items.push(tripItem(trip, false)));

  document.getElementById("tripsList").replaceChildren(...items);
  document.getElementById("tripsEmpty").classList.toggle("hidden", items.length > 0);
}

// Built with textContent rather than innerHTML, so a trip title can't inject HTML.
function tripItem(trip, unsaved) {
  const running = Boolean(activeRun) && (unsaved || activeRun.threadId === trip.thread_id);
  const active = unsaved
    ? activeRun.onScreen
    : trip.thread_id === currentThreadId && !(activeRun && activeRun.onScreen && !running);

  const item = document.createElement("li");
  item.className = "trip-item";
  item.classList.toggle("active", active);

  const open = document.createElement("button");
  open.className = "trip-open";
  open.onclick = () => (unsaved ? showRunOnScreen() : openTrip(trip.thread_id));

  const title = document.createElement("span");
  title.className = "trip-title";
  title.textContent = trip.title;

  const detail = document.createElement("span");
  if (running) {
    detail.className = "trip-running";
    detail.textContent = "Planning…";
  } else {
    detail.className = "trip-date";
    detail.textContent = new Date(trip.updated_at).toLocaleDateString();
  }

  open.append(title, detail);
  item.append(open);

  if (!unsaved) {
    const menuButton = document.createElement("button");
    menuButton.className = "trip-menu-btn";
    menuButton.textContent = "⋯";
    menuButton.setAttribute("aria-label", `Options for ${trip.title}`);
    menuButton.onclick = (event) => {
      event.stopPropagation();
      toggleTripMenu(item, trip, running);
    };
    item.append(menuButton);
  }

  return item;
}

function closeTripMenus() {
  document.querySelectorAll(".trip-menu").forEach((menu) => menu.remove());
}

function toggleTripMenu(item, trip, running) {
  const alreadyOpen = item.querySelector(".trip-menu");
  closeTripMenus();
  if (alreadyOpen) {
    return;
  }

  const menu = document.createElement("div");
  menu.className = "trip-menu";

  const rename = document.createElement("button");
  rename.textContent = "Rename";
  rename.onclick = () => startRename(item, trip);
  menu.append(rename);

  // A trip being planned can't be deleted until it stops.
  if (!running) {
    const remove = document.createElement("button");
    remove.className = "menu-danger";
    remove.textContent = "Delete";
    remove.onclick = () => confirmDelete(trip);
    menu.append(remove);
  }

  item.append(menu);
}

document.addEventListener("click", (event) => {
  if (!event.target.closest(".trip-menu")) {
    closeTripMenus();
  }
});

function startRename(item, trip) {
  closeTripMenus();

  const form = document.createElement("form");
  form.className = "trip-rename";

  const input = document.createElement("input");
  input.className = "input";
  input.value = trip.title;
  input.maxLength = 80;
  input.setAttribute("aria-label", "Trip name");

  const hint = document.createElement("span");
  hint.className = "trip-rename-hint";
  hint.textContent = "Enter to save · Esc to cancel";

  form.append(input, hint);
  item.replaceChildren(form);
  input.focus();
  input.select();

  let finished = false;
  const cancel = () => {
    if (!finished) {
      finished = true;
      renderTrips();
    }
  };

  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      cancel();
    }
  });
  input.addEventListener("blur", cancel);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const title = input.value.trim();
    if (!title || title === trip.title) {
      cancel();
      return;
    }
    finished = true;
    await renameTrip(trip.thread_id, title);
  });
}

// The list updates as soon as the server agrees, then reloads from the server
// (the database is a few round trips away, which takes seconds).
async function renameTrip(threadId, title) {
  try {
    const response = await apiFetch(`/api/trips/${encodeURIComponent(threadId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    const data = await response.json();
    if (!response.ok || !data.success) {
      throw new Error(data.error || "Could not rename the trip.");
    }
    savedTrips = savedTrips.map((t) => (t.thread_id === threadId ? { ...t, title: data.title } : t));
    renderTrips();
  } catch (error) {
    showToast(error.message, { isError: true });
  }
  await loadTrips();
}

function confirmDelete(trip) {
  closeTripMenus();

  const dialog = document.getElementById("confirmDialog");
  document.getElementById("confirmText").textContent =
    `“${trip.title}” and its plan will be deleted. This can't be undone.`;

  dialog.returnValue = "";
  dialog.addEventListener(
    "close",
    () => {
      if (dialog.returnValue === "confirm") {
        deleteTrip(trip.thread_id);
      }
    },
    { once: true },
  );
  dialog.showModal();
}

async function deleteTrip(threadId) {
  try {
    const response = await apiFetch(`/api/trips/${encodeURIComponent(threadId)}`, {
      method: "DELETE",
    });
    const data = await response.json();
    if (!response.ok || !data.success) {
      throw new Error(data.error || "Could not delete the trip.");
    }

    savedTrips = savedTrips.filter((t) => t.thread_id !== threadId);

    if (threadId === currentThreadId) {
      newTrip();
    }
    showToast("Trip deleted.");
  } catch (error) {
    showToast(error.message, { isError: true });
  }
  await loadTrips();
}

// =========================
// Switching between trips
// =========================

const NEW_TRIP_TEXT = {
  title: "Where do you want to go?",
  hint: "Example: Plan a 3 day trip from Melbourne to Tokyo leaving 20 November.",
  placeholder: "Where from, where to, when, and for how long...",
  button: "Generate plan",
};

const FEEDBACK_TEXT = {
  title: "Want to change anything?",
  hint: 'Tell the planner what to change, e.g. "make it 5 days" or "add more food spots on day 2".',
  placeholder: "Your feedback on the plan...",
  button: "Revise plan",
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
}

// Takes the running plan off screen; it keeps going in the background.
function leaveRunScreen() {
  if (activeRun) {
    activeRun.onScreen = false;
  }
  document.getElementById("progressPanel").classList.add("hidden");
  refreshControls();
}

function openTrip(threadId) {
  hideError();
  closeTripMenus();

  if (activeRun && activeRun.threadId === threadId) {
    showRunOnScreen();
    return;
  }

  leaveRunScreen();
  currentThreadId = threadId;
  localStorage.setItem("travel_thread_id", threadId);
  hideFlightChoice();
  restorePlan();
  renderTrips();
}

function newTrip() {
  leaveRunScreen();
  hideFlightChoice();
  currentThreadId = null;
  latestAnswerMarkdown = "";
  localStorage.removeItem("travel_thread_id");

  document.getElementById("resultSection").classList.add("hidden");
  document.getElementById("userInput").value = "";
  hideError();
  setMode(false);
  renderTrips();
}

// Brings the running plan back on screen: its progress, and the text so far.
function showRunOnScreen() {
  const run = activeRun;
  if (!run) {
    return;
  }

  hideError();
  hideFlightChoice();
  run.onScreen = true;
  currentThreadId = run.threadId;
  document.getElementById("progressPanel").classList.remove("hidden");

  if (run.liveText) {
    renderLivePlan(run.liveText);
  } else if (!run.isNew && run.threadId) {
    showSavedPlanOnly(run.threadId);
  } else {
    document.getElementById("resultSection").classList.add("hidden");
  }

  setMode(!run.isNew);
  refreshControls();
  renderTrips();
}

// The saved plan of a trip, without its flight cards (used while it's re-running).
async function showSavedPlanOnly(threadId) {
  try {
    const response = await apiFetch(`/api/travel/${encodeURIComponent(threadId)}`);
    const data = await response.json();
    if (response.ok && data.answer && currentThreadId === threadId && !activeRun?.liveText) {
      showResult(data.answer, threadId, undefined, false);
    }
  } catch (error) {
    // Not critical: the plan appears once the model starts writing.
  }
}

async function restorePlan() {
  const threadId = currentThreadId;

  if (!threadId) {
    setMode(false);
    return;
  }

  try {
    const response = await apiFetch(`/api/travel/${encodeURIComponent(threadId)}`);
    const data = await response.json();

    // The user opened something else while this was loading.
    if (threadId !== currentThreadId || (activeRun && activeRun.onScreen)) {
      return;
    }

    if (response.ok && data.success) {
      // A trip paused on a flight choice has options but no plan yet.
      if (data.flight_options) {
        showFlightChoice(data.flight_options);
      } else {
        hideFlightChoice();
      }

      if (data.answer) {
        showResult(data.answer, threadId);
      } else {
        document.getElementById("resultSection").classList.add("hidden");
      }
      setMode(Boolean(data.answer));
      renderTrips();
      return;
    }
  } catch (error) {
    // Fall through and start a new trip.
    if (!authToken) {
      return;
    }
  }

  if (threadId === currentThreadId) {
    newTrip();
  }
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
  renderMarkdown(document.getElementById("resultBox"), answer);

  const routeLabel = ROUTE_LABELS[route] ? `${ROUTE_LABELS[route]} · ` : "";
  document.getElementById("threadInfo").textContent = `${routeLabel}Trip ID: ${threadId}`;

  resultSection.classList.remove("hidden");

  if (scroll) {
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
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
  const anyRunning = [...steps.values()].some((step) => step.status === "running");

  if (now - lastDataAt > CONNECTION_LOST_MS) {
    showProgressWarning(
      "No response from the server. The connection may have dropped; press Stop and try again.",
    );
  } else if (anyRunning && now - lastProgressAt > SLOW_PROGRESS_MS) {
    showProgressWarning("This is taking longer than usual. You can keep waiting or press Stop.");
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
  if (activeRun) {
    activeRun.controller.abort();
  }
}

function renderLivePlan(markdown) {
  const resultSection = document.getElementById("resultSection");
  const firstRender = resultSection.classList.contains("hidden");

  renderMarkdown(document.getElementById("resultBox"), markdown);
  document.getElementById("threadInfo").textContent = "Writing...";
  resultSection.classList.remove("hidden");

  if (firstRender) {
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

// Shows the plan while the model writes it, re-rendered at most every 200 ms,
// and only while the running trip is on screen.
function scheduleLiveRender() {
  if (liveRenderTimer) {
    return;
  }

  liveRenderTimer = setTimeout(() => {
    liveRenderTimer = null;
    if (activeRun && activeRun.onScreen) {
      renderLivePlan(activeRun.liveText);
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
    (Date.parse((arrival || "").split(" ")[0]) - Date.parse((departure || "").split(" ")[0])) /
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
  const stops = option.stops === 0 ? "Direct" : `${option.stops} stop${option.stops > 1 ? "s" : ""}`;
  details.textContent = `${option.duration} · ${stops}`;

  const button = document.createElement("button");
  button.className = "button button-primary button-small";
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

function tripTitle(threadId) {
  const trip = savedTrips.find((t) => t.thread_id === threadId);
  return trip ? trip.title : "your trip";
}

// Streams one run: a new request, feedback, or the answer to a flight choice.
// onAccepted runs once the server has taken the request (plan done or paused),
// if the trip is still on screen.
async function runPlan(body, onAccepted = () => {}) {
  if (activeRun) {
    showError(RUN_IN_PROGRESS);
    return;
  }

  const run = {
    threadId: body.thread_id || null,
    isNew: !body.thread_id,
    title: body.message || tripTitle(body.thread_id),
    controller: new AbortController(),
    liveText: "",
    onScreen: true,
  };
  activeRun = run;

  hideError();
  hideFlightChoice();
  startProgress();
  refreshControls();
  renderTrips();

  let finished = false;

  // Drops the trip from the screen if it was being shown, and announces it otherwise.
  const announce = (text, label) => {
    const threadId = run.threadId;
    showToast(text, { actionLabel: label, onAction: () => openTrip(threadId) });
  };

  try {
    const response = await apiFetch("/api/travel/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: run.controller.signal,
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || "Something went wrong.");
    }

    await readEvents(response, (event) => {
      lastProgressAt = Date.now();

      switch (event.type) {
        case "start":
          run.threadId = event.thread_id;
          if (event.title && run.isNew) {
            run.title = event.title;
          }
          if (run.onScreen) {
            currentThreadId = event.thread_id;
          }
          renderTrips();
          break;
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
          if (!run.liveText) {
            steps.forEach((step, node) => {
              if (step.status === "running") {
                setStepMessage(node, "");
              }
            });
          }
          run.liveText += event.text;
          scheduleLiveRender();
          break;
        case "choose":
          // The run is paused and saved; it continues when a flight is picked.
          finished = true;
          endProgress("Pick a flight to continue");

          if (run.onScreen) {
            localStorage.setItem("travel_thread_id", event.thread_id);
            showFlightChoice(event.options);
            onAccepted();
          } else {
            announce(`Pick a flight for “${run.title}”.`, "Open");
          }
          loadTrips();
          break;
        case "done":
          finished = true;
          endProgress("Plan ready");

          if (run.onScreen) {
            localStorage.setItem("travel_thread_id", event.thread_id);
            showResult(event.answer, event.thread_id, event.route, false);
            onAccepted();
            setMode(true);
          } else {
            announce(`Your plan for “${run.title}” is ready.`, "Open");
          }
          loadTrips();
          break;
        case "error":
          throw new Error(event.message);
      }
    });

    if (!finished) {
      throw new Error("The connection closed before the plan was finished. Please try again.");
    }
  } catch (error) {
    const stopped = error.name === "AbortError";
    endProgress(stopped ? "Stopped" : "Planning failed");

    if (run.onScreen) {
      if (!stopped) {
        showError(error.message);
      }
      // Drop the half-written plan and show the saved state (a plan, or the
      // flight cards of a paused trip). A new trip has nothing saved.
      if (run.isNew) {
        currentThreadId = null;
        document.getElementById("resultSection").classList.add("hidden");
      } else {
        activeRun = null;
        restorePlan();
      }
    } else if (!stopped) {
      showToast(`Planning “${run.title}” failed: ${error.message}`, { isError: true });
    }
  } finally {
    if (activeRun === run) {
      activeRun = null;
    }
    refreshControls();
    renderTrips();
  }
}

function copyResult() {
  const text = document.getElementById("resultBox").innerText;

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
      backgroundColor: "#fbf8f2",
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
  if (event.key === "Escape") {
    closeTripMenus();
  }
});

if (authToken) {
  showApp();
} else {
  showAuth();
}
