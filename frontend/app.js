// ---- Cursor-following glow ----
document.addEventListener("mousemove", (e) => {
  document.documentElement.style.setProperty("--x", `${e.clientX}px`);
  document.documentElement.style.setProperty("--y", `${e.clientY}px`);
});

// ---- Config ----
const API_BASE = ""; // same-origin, since FastAPI serves this file too
const SESSION_KEY = "trendly_session_id";

// ---- Elements ----
const messagesEl = document.getElementById("messages");
const typingEl = document.getElementById("typingIndicator");
const formEl = document.getElementById("chatForm");
const inputEl = document.getElementById("chatInput");
const newChatBtn = document.getElementById("newChatBtn");
const filterIconEl = document.getElementById("filter-icon");
const quickChipsEl = document.getElementById("quickChips");
const landingEl = document.getElementById("landing");
const chatContainerEl = document.getElementById("chatContainer");
const startChattingBtn = document.getElementById("startChattingBtn");
const landingQuickCardsEl = document.getElementById("landingQuickCards");

// ---- Session handling ----
function getSessionId() {
  return localStorage.getItem(SESSION_KEY);
}
function setSessionId(id) {
  localStorage.setItem(SESSION_KEY, id);
}
function clearSession() {
  localStorage.removeItem(SESSION_KEY);
}

// ---- Rendering ----
function appendMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function appendError(text) {
  const div = document.createElement("div");
  div.className = "message error";
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setTyping(isTyping) {
  typingEl.classList.toggle("hidden", !isTyping);
  if (isTyping) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function resetChat() {
  messagesEl.innerHTML = "";
  appendMessage("Hi! I'm the Trendly support assistant. How can I help you today?", "assistant");
  clearSession();
}

// ---- Sending messages ----
async function sendMessage(text) {
  if (!text.trim()) return;

  appendMessage(text, "user");
  inputEl.value = "";
  setTyping(true);

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: getSessionId() || null,
      }),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    setSessionId(data.session_id);
    appendMessage(data.reply, "assistant");
  } catch (err) {
    appendError(`Something went wrong: ${err.message}`);
  } finally {
    setTyping(false);
  }
}

// ---- Event wiring ----
formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(inputEl.value);
});

newChatBtn.addEventListener("click", () => {
  resetChat();
});

filterIconEl.addEventListener("click", () => {
  resetChat();
});

quickChipsEl.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  sendMessage(chip.dataset.prompt);
});

// ---- Landing -> chat transition ----
function enterChat(prefillPrompt) {
  landingEl.classList.add("fade-out");
  landingEl.addEventListener(
    "animationend",
    () => {
      landingEl.classList.add("hidden");
      landingEl.classList.remove("fade-out");

      chatContainerEl.classList.remove("hidden");
      chatContainerEl.classList.add("fade-in");
      chatContainerEl.addEventListener(
        "animationend",
        () => chatContainerEl.classList.remove("fade-in"),
        { once: true }
      );

      if (prefillPrompt) sendMessage(prefillPrompt);
    },
    { once: true }
  );
}

startChattingBtn.addEventListener("click", () => enterChat());

landingQuickCardsEl.addEventListener("click", (e) => {
  const card = e.target.closest(".landing-card");
  if (!card) return;
  enterChat(card.dataset.prompt);
});

// ---- Health check on load (optional, silent) ----
fetch(`${API_BASE}/health`).catch(() => {
  // Backend not reachable yet; chat requests will surface the real error.
});