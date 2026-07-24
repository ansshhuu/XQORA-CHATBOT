
 
(function () {
  "use strict";
  var THEME = {
    teal: "#14b8a6",
    tealDark: "#0d9488",
    navy: "#0a1e3f",
    textDark: "#0f172a",
    panelBg: "#f8fafc",
    white: "#ffffff",
    radius: "14px",
    radiusSm: "10px",
    shadow: "0 10px 30px rgba(10, 30, 63, 0.18)",
    bubbleShadow: "0 6px 18px rgba(20, 184, 166, 0.45)",
  };

  var currentScript = document.currentScript;
  var API_URL =
    (currentScript && currentScript.getAttribute("data-api-url")) || "/chat";


  var sessionId = null;
  if (window.crypto && window.crypto.randomUUID) {
    sessionId = window.crypto.randomUUID();
  } else {
    sessionId = "xqora-" + Date.now() + "-" + Math.random().toString(16).slice(2);
  }

  var isOpen = false;
  var isSending = false;


  var style = document.createElement("style");
  style.textContent =
    "#xqora-chat-bubble {" +
    "position: fixed; bottom: 24px; right: 24px; width: 60px; height: 60px;" +
    "border-radius: 50%; background: " + THEME.teal + ";" +
    "box-shadow: " + THEME.bubbleShadow + ";" +
    "display: flex; align-items: center; justify-content: center;" +
    "cursor: pointer; z-index: 2147483000; border: none;" +
    "transition: transform 0.15s ease, box-shadow 0.15s ease;" +
    "}" +
    "#xqora-chat-bubble:hover { transform: scale(1.08); box-shadow: 0 8px 22px rgba(20, 184, 166, 0.6); }" +
    "#xqora-chat-bubble svg { width: 28px; height: 28px; }" +
    "#xqora-chat-panel {" +
    "position: fixed; bottom: 96px; right: 24px; width: 400px; max-width: calc(100vw - 32px);" +
    "height: 580px; max-height: calc(100vh - 140px);" +
    "background: " + THEME.panelBg + ";" +
    "border-radius: " + THEME.radius + ";" +
    "box-shadow: " + THEME.shadow + ";" +
    "display: none; flex-direction: column; overflow: hidden;" +
    "z-index: 2147483000; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;" +
    "}" +
    "#xqora-chat-panel.xqora-open { display: flex; }" +
    "#xqora-chat-header {" +
    "background: " + THEME.navy + "; color: " + THEME.white + ";" +
    "padding: 14px 16px; display: flex; align-items: center; gap: 10px;" +
    "flex-shrink: 0;" +
    "}" +
    "#xqora-chat-header .xqora-logo {" +
    "width: 34px; height: 34px; border-radius: 50%; background: " + THEME.teal + ";" +
    "display: flex; align-items: center; justify-content: center;" +
    "font-weight: 700; font-size: 14px; color: " + THEME.white + "; flex-shrink: 0;" +
    "}" +
    "#xqora-chat-header .xqora-header-text { display: flex; flex-direction: column; gap: 2px; min-width: 0; }" +
    "#xqora-chat-header .xqora-title {" +
    "font-weight: 600; font-size: 14px; line-height: 1.2; color: " + THEME.white + ";" +
    "}" +
    "#xqora-chat-messages {" +
    "flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 10px;" +
    "background: " + THEME.panelBg + ";" +
    "}" +
    ".xqora-msg-row { display: flex; }" +
    ".xqora-msg-row.xqora-user { justify-content: flex-end; }" +
    ".xqora-msg-row.xqora-bot { justify-content: flex-start; }" +
    ".xqora-bubble {" +
    "max-width: 78%; padding: 9px 13px; border-radius: " + THEME.radiusSm + ";" +
    "font-size: 13.5px; line-height: 1.45; white-space: pre-wrap; word-wrap: break-word;" +
    "}" +
    ".xqora-bubble.xqora-user { background: " + THEME.teal + "; color: " + THEME.white + "; border-bottom-right-radius: 3px; }" +
    ".xqora-bubble.xqora-bot {" +
    "background: " + THEME.white + "; color: " + THEME.textDark + ";" +
    "border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);" +
    "border-bottom-left-radius: 3px;" +
    "}" +
    "#xqora-chat-inputbar {" +
    "display: flex; gap: 8px; padding: 10px; background: " + THEME.white + ";" +
    "border-top: 1px solid #e2e8f0; flex-shrink: 0;" +
    "}" +
    "#xqora-chat-input {" +
    "flex: 1; border: 1.5px solid #e2e8f0; border-radius: 999px;" +
    "padding: 9px 14px; font-size: 13.5px; outline: none; background: " + THEME.white + ";" +
    "color: " + THEME.textDark + "; transition: border-color 0.15s ease, box-shadow 0.15s ease;" +
    "}" +
    "#xqora-chat-input:focus {" +
    "border-color: " + THEME.teal + "; box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.18);" +
    "}" +
    "#xqora-chat-send {" +
    "background: " + THEME.teal + "; color: " + THEME.white + "; border: none;" +
    "border-radius: 999px; width: 38px; height: 38px; flex-shrink: 0; cursor: pointer;" +
    "display: flex; align-items: center; justify-content: center;" +
    "transition: background 0.15s ease, opacity 0.15s ease;" +
    "}" +
    "#xqora-chat-send:hover { background: " + THEME.tealDark + "; }" +
    "#xqora-chat-send:disabled { opacity: 0.5; cursor: not-allowed; }" +
    "#xqora-chat-send svg { width: 16px; height: 16px; }" +
    ".xqora-typing { display: flex; gap: 4px; padding: 4px 2px; }" +
    ".xqora-typing span {" +
    "width: 6px; height: 6px; border-radius: 50%; background: #94a3b8;" +
    "animation: xqora-bounce 1.2s infinite ease-in-out;" +
    "}" +
    ".xqora-typing span:nth-child(2) { animation-delay: 0.15s; }" +
    ".xqora-typing span:nth-child(3) { animation-delay: 0.3s; }" +
    "@keyframes xqora-bounce { 0%, 60%, 100% { transform: translateY(0); opacity: 0.5; } 30% { transform: translateY(-4px); opacity: 1; } }" +
    "@media (max-width: 420px) {" +
    "#xqora-chat-panel { left: 12px; right: 12px; width: auto; bottom: 88px; }" +
    "#xqora-chat-bubble { right: 16px; bottom: 16px; }" +
    "}";
  document.head.appendChild(style);

  var CHAT_ICON_SVG =
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">' +
    '<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" stroke="#ffffff" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>' +
    '<circle cx="8" cy="12" r="1" fill="#ffffff"/>' +
    '<circle cx="12" cy="12" r="1" fill="#ffffff"/>' +
    '<circle cx="16" cy="12" r="1" fill="#ffffff"/>' +
    "</svg>";
  var CLOSE_ICON_SVG =
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">' +
    '<path d="M6 6l12 12M18 6L6 18" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>' +
    "</svg>";

  var bubble = document.createElement("button");
  bubble.id = "xqora-chat-bubble";
  bubble.setAttribute("aria-label", "Open XQORA chat");
  bubble.innerHTML = CHAT_ICON_SVG;

  var panel = document.createElement("div");
  panel.id = "xqora-chat-panel";
  panel.innerHTML =
    '<div id="xqora-chat-header">' +
    '<div class="xqora-logo">X</div>' +
    '<div class="xqora-header-text">' +
    '<div class="xqora-title">XQORA Assistant</div>' +
    "</div>" +
    "</div>" +
    '<div id="xqora-chat-messages"></div>' +
    '<div id="xqora-chat-inputbar">' +
    '<input id="xqora-chat-input" type="text" placeholder="Type a message..." autocomplete="off" />' +
    '<button id="xqora-chat-send" aria-label="Send message">' +
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">' +
    '<path d="M4 20l16-8L4 4v6l10 2-10 2v6z" fill="#ffffff"/>' +
    "</svg>" +
    "</button>" +
    "</div>";

  document.body.appendChild(bubble);
  document.body.appendChild(panel);

  var messagesEl = panel.querySelector("#xqora-chat-messages");
  var inputEl = panel.querySelector("#xqora-chat-input");
  var sendBtn = panel.querySelector("#xqora-chat-send");

  function addMessage(text, sender) {
    var row = document.createElement("div");
    row.className = "xqora-msg-row " + (sender === "user" ? "xqora-user" : "xqora-bot");
    var bubbleEl = document.createElement("div");
    bubbleEl.className = "xqora-bubble " + (sender === "user" ? "xqora-user" : "xqora-bot");
    bubbleEl.textContent = text;
    row.appendChild(bubbleEl);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return row;
  }

  function addTypingIndicator() {
    var row = document.createElement("div");
    row.className = "xqora-msg-row xqora-bot";
    row.id = "xqora-typing-row";
    var bubbleEl = document.createElement("div");
    bubbleEl.className = "xqora-bubble xqora-bot";
    bubbleEl.innerHTML = '<div class="xqora-typing"><span></span><span></span><span></span></div>';
    row.appendChild(bubbleEl);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function removeTypingIndicator() {
    var row = document.getElementById("xqora-typing-row");
    if (row) row.remove();
  }

  function updateBubbleIcon() {
    bubble.innerHTML = isOpen ? CLOSE_ICON_SVG : CHAT_ICON_SVG;
    bubble.setAttribute("aria-label", isOpen ? "Close XQORA chat" : "Open XQORA chat");
  }

  function openPanel() {
    isOpen = true;
    panel.classList.add("xqora-open");
    updateBubbleIcon();
    if (messagesEl.children.length === 0) {
      addMessage("Hey there! What can I help you with today?", "bot");
    }
    inputEl.focus();
  }

  function closePanel() {
    isOpen = false;
    panel.classList.remove("xqora-open");
    updateBubbleIcon();
  }

  bubble.addEventListener("click", function () {
    isOpen ? closePanel() : openPanel();
  });

  function sendMessage() {
    var text = inputEl.value.trim();
    if (!text || isSending) return;

    addMessage(text, "user");
    inputEl.value = "";
    isSending = true;
    sendBtn.disabled = true;
    addTypingIndicator();

    fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    })
      .then(function (res) {
        return res.json().then(function (data) {
          return { ok: res.ok, data: data };
        });
      })
      .then(function (result) {
        removeTypingIndicator();
        if (result.ok && result.data && result.data.reply) {
          if (result.data.session_id) sessionId = result.data.session_id;
          addMessage(result.data.reply, "bot");
        } else {
          addMessage("Sorry, something went wrong. Please try again in a moment.", "bot");
        }
      })
      .catch(function () {
        removeTypingIndicator();
        addMessage("Sorry, I couldn't reach the server. Please check your connection and try again.", "bot");
      })
      .finally(function () {
        isSending = false;
        sendBtn.disabled = false;
        inputEl.focus();
      });
  }

  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.isComposing && e.keyCode !== 229) sendMessage();
  });
})();
