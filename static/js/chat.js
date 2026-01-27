const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const messages = document.getElementById("messages");

// Base URL automatically includes protocol + host + port
const API_BASE = window.location.origin;

// Function to add a message to the chat window
function addMessage(text, type) {
  const div = document.createElement("div");
  div.className = `message ${type}`;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

// Bot message with translate button and animation
function addBotMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message bot";

  const content = document.createElement("div");
  content.className = "bot-text";

  const textSpan = document.createElement("span");
  content.appendChild(textSpan);

  // Check for image URL in text
  const imgMatch = text.match(/(\/static\/graphs\/[^\s]+\.png)/);
  if (imgMatch) {
    const cleanText = text.replace(imgMatch[0], "").trim();
    textSpan.textContent = cleanText;

    const img = document.createElement("img");
    img.src = imgMatch[0];
    img.className = "chat-graph";
    img.style.maxWidth = "100%";
    img.style.borderRadius = "8px";
    img.style.marginTop = "10px";
    img.alt = "Data Visualization";
    content.appendChild(img);
  } else {
    textSpan.textContent = text;
  }

  const btn = document.createElement("button");
  btn.className = "translate-btn";
  btn.textContent = "Translate";

  let currentLang = "en"; // default language

  btn.onclick = async () => {
    btn.disabled = true;
    btn.textContent = "Translating…";

    try {
      const res = await fetch(`${API_BASE}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: textSpan.textContent,
          target_lang: currentLang === "en" ? "Arabic" : "English"
        })
      });

      if (!res.ok) throw new Error("Translation API error");

      const data = await res.json();

      // Strip any graph URLs from the translation to keep text clean
      const graphRegex = /\/static\/graphs\/[^\s]+\.png/gi;
      textSpan.textContent = data.translation.replace(graphRegex, "").trim();

      currentLang = currentLang === "en" ? "ar" : "en";
      btn.textContent = currentLang === "en" ? "Translate" : "Original";

    } catch (err) {
      alert("Translation failed!");
      console.error(err);
    } finally {
      btn.disabled = false;
    }
  };

  wrapper.appendChild(content);
  wrapper.appendChild(btn);
  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
}

// Function to show bot typing indicator
function addBotTyping() {
  const wrapper = document.createElement("div");
  wrapper.className = "message bot typing";
  wrapper.textContent = "Co-op Magic AI Assistant is typing";

  // Add animated dots
  const dots = document.createElement("span");
  dots.className = "dots";
  dots.textContent = "...";
  wrapper.appendChild(dots);

  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;

  return wrapper; // Return so we can remove it later
}

// Initial greeting when chatbot loads
window.addEventListener("DOMContentLoaded", () => {
  addBotMessage("👋 Hello! I'm Co-op Magic AI Assistant. Ask me anything about cooperatives in South Sudan. I can translate both English and Arabic.");
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const message = input.value.trim();
  if (!message) return;

  addMessage(message, "user");
  input.value = "";

  // Show typing indicator
  const typingElem = addBotTyping();

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    if (!res.ok) {
      throw new Error("Server error");
    }

    const data = await res.json();

    // Remove typing indicator
    typingElem.remove();

    addBotMessage(data.reply);
  } catch {
    addMessage("Unable to reach the server.", "bot");
  }
});
