
 
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
  var FEEDBACK_SUBMIT_URL =
    (currentScript && currentScript.getAttribute("data-feedback-submit-url")) || "/feedback/submit";
  var FEEDBACK_CANCEL_URL =
    (currentScript && currentScript.getAttribute("data-feedback-cancel-url")) || "/feedback/cancel";
  // Served by Vercel directly from /public at the site root, so this is
  // root-relative (no /static prefix - nothing mounts these files via
  // FastAPI anymore).
  var AVATAR_URL = "/bot-avatar.png";

  // Must match app/prompt.py's FEEDBACK_ASK exactly - the server still
  // appends this literal text to the reply (backend logic is unchanged),
  // the widget just strips it back off and renders the interactive card
  // in its place when awaiting_feedback comes back true.
  var FEEDBACK_ASK_TEXT = "Before you go, how'd this chat go? (1-5, or just tell me)";

  var GREETING_TEXT = "Hey there! What can I help you with today?";

  var QUICK_REPLIES = [
    "I want to hire you",
    "Tell me your services",
    "What is XQORA?",
  ];

  var FEEDBACK_FACES = [
    { rating: "happy", emoji: "😊", label: "Happy" },
    { rating: "ok", emoji: "😐", label: "Ok" },
    { rating: "sad", emoji: "🙁", label: "Sad" },
  ];

  var FEEDBACK_REASONS = [
    { reason: "bad_app_experience", label: "Bad app experience" },
    { reason: "didnt_understand_query", label: "Didn't understand my query" },
    { reason: "slow_response", label: "Slow response" },
    { reason: "other", label: "Other" },
  ];

  var SESSION_ID_STORAGE_KEY = "xqora_session_id";

  // sessionStorage (not localStorage) so history/session survive a reload
  // but clear on tab close, matching "same session" - wrapped since it can
  // throw in sandboxed/embedded contexts (privacy mode, third-party iframe).
  function _storageGet(key) {
    try {
      return sessionStorage.getItem(key);
    } catch (e) {
      return null;
    }
  }
  function _storageSet(key, value) {
    try {
      sessionStorage.setItem(key, value);
    } catch (e) {}
  }
  function _storageRemove(key) {
    try {
      sessionStorage.removeItem(key);
    } catch (e) {}
  }

  // Reused across reloads (instead of a fresh UUID every load) so the chat
  // history restored below still maps to the same backend session_id - the
  // backend already accepts and reuses whatever session_id the client sends
  // (see app/routes/chat.py), it just needs the client to actually persist
  // and resend the same one.
  var sessionId = _storageGet(SESSION_ID_STORAGE_KEY);
  if (!sessionId) {
    if (window.crypto && window.crypto.randomUUID) {
      sessionId = window.crypto.randomUUID();
    } else {
      sessionId = "xqora-" + Date.now() + "-" + Math.random().toString(16).slice(2);
    }
    _storageSet(SESSION_ID_STORAGE_KEY, sessionId);
  }

  function _historyKey() {
    return "chat_history_" + sessionId;
  }
  function _feedbackPendingKey() {
    return "chat_feedback_pending_" + sessionId;
  }

  function _loadHistory() {
    var raw = _storageGet(_historyKey());
    if (!raw) return [];
    try {
      var parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      return [];
    }
  }

  function _appendHistoryEntry(sender, text) {
    var history = _loadHistory();
    history.push({ sender: sender, text: text, timestamp: new Date().toISOString() });
    _storageSet(_historyKey(), JSON.stringify(history));
  }

  var isOpen = false;
  var isSending = false;
  var activeFeedbackCard = null;


  var style = document.createElement("style");
  style.textContent =
    "#xqora-chat-bubble {" +
    "position: fixed; bottom: 24px; right: 24px; width: 60px; height: 60px;" +
    "border-radius: 50%; background: " + THEME.teal + ";" +
    "box-shadow: " + THEME.bubbleShadow + ";" +
    "display: flex; align-items: center; justify-content: center;" +
    "cursor: pointer; z-index: 2147483000; border: none; padding: 0;" +
    "overflow: hidden; box-sizing: border-box;" +
    "transition: transform 0.15s ease, box-shadow 0.15s ease;" +
    "}" +
    "#xqora-chat-bubble:hover { transform: scale(1.08); box-shadow: 0 8px 22px rgba(20, 184, 166, 0.6); }" +
    "#xqora-chat-bubble.xqora-bubble-open { background: " + THEME.navy + "; }" +
    "#xqora-chat-bubble svg { width: 28px; height: 28px; }" +
    "#xqora-chat-bubble img {" +
    "width: 100%; height: 100%; border-radius: 50%;" +
    "object-fit: cover; object-position: center; display: block;" +
    "}" +
    "#xqora-chat-panel {" +
    "position: fixed; bottom: 96px; right: 24px; width: 440px; max-width: calc(100vw - 32px);" +
    "height: 680px; max-height: calc(100vh - 120px);" +
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
    "flex-shrink: 0; box-shadow: 0 2px 8px rgba(10, 30, 63, 0.25); position: relative; z-index: 1;" +
    "}" +
    "#xqora-chat-header .xqora-logo-wrap {" +
    "position: relative; width: 54px; height: 54px; flex-shrink: 0;" +
    "}" +
    "#xqora-chat-header .xqora-logo {" +
    "width: 100%; height: 100%; border-radius: 50%; display: block;" +
    "object-fit: cover; object-position: center; box-sizing: border-box;" +
    "border: 2.5px solid " + THEME.white + ";" +
    "}" +
    "#xqora-chat-header .xqora-status-dot {" +
    "position: absolute; bottom: 0; right: 0; width: 14px; height: 14px;" +
    "border-radius: 50%; background: " + THEME.teal + ";" +
    "border: 2px solid " + THEME.white + "; box-sizing: border-box;" +
    "}" +
    "#xqora-chat-header .xqora-header-text { display: flex; flex-direction: column; gap: 2px; min-width: 0; }" +
    "#xqora-chat-header .xqora-title {" +
    "font-weight: 600; font-size: 14px; line-height: 1.2; color: " + THEME.white + ";" +
    "}" +
    "#xqora-chat-header .xqora-subtitle {" +
    "font-weight: 500; font-size: 11.5px; line-height: 1.2; color: " + THEME.teal + ";" +
    "}" +
    "#xqora-chat-messages {" +
    "flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 10px;" +
    "background: " + THEME.panelBg + ";" +
    "}" +
    "@keyframes xqora-msg-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }" +
    ".xqora-msg-row { display: flex; animation: xqora-msg-in 0.2s ease; }" +
    ".xqora-msg-row.xqora-user { justify-content: flex-end; }" +
    ".xqora-msg-row.xqora-bot { justify-content: flex-start; align-items: center; gap: 10px; }" +
    ".xqora-bubble {" +
    "max-width: 78%; padding: 9px 13px; border-radius: " + THEME.radiusSm + ";" +
    "font-size: 13.5px; line-height: 1.45; white-space: pre-wrap; word-wrap: break-word;" +
    "box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);" +
    "}" +
    ".xqora-bubble.xqora-user {" +
    "background: " + THEME.teal + "; color: " + THEME.white + "; border-bottom-right-radius: 3px;" +
    "box-shadow: 0 3px 10px rgba(20, 184, 166, 0.25);" +
    "}" +
    ".xqora-bubble.xqora-bot {" +
    "background: " + THEME.white + "; color: " + THEME.textDark + ";" +
    "border: 1px solid #e2e8f0;" +
    "border-bottom-left-radius: 3px;" +
    "}" +
    ".xqora-bot-avatar {" +
    "width: 42px; height: 42px; border-radius: 50%; flex-shrink: 0;" +
    "object-fit: cover; object-position: center;" +
    "border: none; box-shadow: none; display: block;" +
    "}" +
    ".xqora-quick-replies-row {" +
    "display: flex; flex-wrap: wrap; gap: 8px; padding-left: 52px;" +
    "animation: xqora-msg-in 0.2s ease;" +
    "}" +
    ".xqora-quick-reply-btn {" +
    "background: " + THEME.white + "; color: " + THEME.teal + ";" +
    "border: 1.5px solid " + THEME.teal + "; border-radius: 999px;" +
    "padding: 7px 14px; font-size: 13px; cursor: pointer;" +
    "box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);" +
    "transition: all 0.15s ease;" +
    "}" +
    ".xqora-quick-reply-btn:hover {" +
    "border-color: " + THEME.tealDark + ";" +
    "box-shadow: 0 4px 10px rgba(20, 184, 166, 0.25);" +
    "transform: translateY(-1px);" +
    "}" +
    ".xqora-feedback-card {" +
    "max-width: 85%; background: " + THEME.white + "; border: 1px solid #e2e8f0;" +
    "border-radius: " + THEME.radiusSm + "; border-bottom-left-radius: 3px;" +
    "box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08); padding: 14px;" +
    "}" +
    ".xqora-feedback-title {" +
    "font-weight: 600; font-size: 14px; color: " + THEME.textDark + "; margin-bottom: 2px;" +
    "}" +
    ".xqora-feedback-subtitle {" +
    "font-size: 12.5px; color: #64748b; margin-bottom: 12px;" +
    "}" +
    ".xqora-feedback-faces { display: flex; gap: 8px; margin-bottom: 4px; }" +
    ".xqora-face-btn {" +
    "flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px;" +
    "padding: 8px 4px; border: 1.5px solid #e2e8f0; border-radius: " + THEME.radiusSm + ";" +
    "background: " + THEME.white + "; cursor: pointer; transition: all 0.15s ease;" +
    "}" +
    ".xqora-face-btn:hover { border-color: " + THEME.teal + "; }" +
    ".xqora-face-btn.xqora-selected {" +
    "border-color: " + THEME.teal + "; background: rgba(20, 184, 166, 0.08);" +
    "box-shadow: 0 0 0 2px rgba(20, 184, 166, 0.15);" +
    "}" +
    ".xqora-face-emoji { font-size: 22px; line-height: 1; }" +
    ".xqora-face-label { font-size: 11px; color: " + THEME.textDark + "; }" +
    ".xqora-feedback-reasons {" +
    "display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px;" +
    "}" +
    ".xqora-feedback-reasons.xqora-hidden { display: none; }" +
    ".xqora-reason-chip {" +
    "background: " + THEME.white + "; color: " + THEME.teal + ";" +
    "border: 1.5px solid " + THEME.teal + "; border-radius: 999px;" +
    "padding: 5px 12px; font-size: 12px; cursor: pointer; transition: all 0.15s ease;" +
    "}" +
    ".xqora-reason-chip.xqora-selected { background: " + THEME.teal + "; color: " + THEME.white + "; }" +
    ".xqora-feedback-actions {" +
    "display: flex; justify-content: flex-end; gap: 8px; margin-top: 14px;" +
    "}" +
    ".xqora-feedback-cancel {" +
    "background: " + THEME.white + "; color: #64748b; border: 1.5px solid #e2e8f0;" +
    "border-radius: 999px; padding: 7px 16px; font-size: 13px; cursor: pointer;" +
    "transition: all 0.15s ease;" +
    "}" +
    ".xqora-feedback-cancel:hover { border-color: #cbd5e1; }" +
    ".xqora-feedback-submit {" +
    "background: " + THEME.teal + "; color: " + THEME.white + "; border: none;" +
    "border-radius: 999px; padding: 7px 18px; font-size: 13px; cursor: pointer;" +
    "transition: background 0.15s ease, opacity 0.15s ease;" +
    "}" +
    ".xqora-feedback-submit:hover:not(:disabled) { background: " + THEME.tealDark + "; }" +
    ".xqora-feedback-submit:disabled { opacity: 0.45; cursor: not-allowed; }" +
    ".xqora-auto-restart-note {" +
    "width: 100%; font-size: 12px; color: #64748b; padding: 0 2px 2px;" +
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
    "box-shadow: 0 2px 6px rgba(20, 184, 166, 0.35);" +
    "transition: background 0.15s ease, opacity 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;" +
    "}" +
    "#xqora-chat-send:hover { background: " + THEME.tealDark + "; transform: scale(1.06); box-shadow: 0 4px 12px rgba(20, 184, 166, 0.5); }" +
    "#xqora-chat-send:active { transform: scale(0.96); }" +
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
    "@media (max-width: 480px) {" +
    "#xqora-chat-panel { left: 12px; right: 12px; width: auto; bottom: 88px; }" +
    "#xqora-chat-bubble { right: 16px; bottom: 16px; }" +
    "}";
  document.head.appendChild(style);

  // Launcher bubble icon (before the widget opens) is the same bot logo used
  // in the header/message avatars, filling the circular button - "#xqora-chat-bubble
  // img" above gives it the object-fit: cover crop, same treatment as
  // ".xqora-logo"/".xqora-bot-avatar". Swapped for CLOSE_ICON_SVG once open.
  var CHAT_ICON_IMG_HTML = '<img src="' + AVATAR_URL + '" alt="XQORA" />';
  var CLOSE_ICON_SVG =
    '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">' +
    '<path d="M6 6l12 12M18 6L6 18" stroke="' + THEME.white + '" stroke-width="2" stroke-linecap="round"/>' +
    "</svg>";

  var bubble = document.createElement("button");
  bubble.id = "xqora-chat-bubble";
  bubble.setAttribute("aria-label", "Open XQORA chat");
  bubble.innerHTML = CHAT_ICON_IMG_HTML;

  var panel = document.createElement("div");
  panel.id = "xqora-chat-panel";
  panel.innerHTML =
    '<div id="xqora-chat-header">' +
    '<div class="xqora-logo-wrap">' +
    '<img class="xqora-logo" src="' + AVATAR_URL + '" alt="XQORA" />' +
    '<span class="xqora-status-dot"></span>' +
    "</div>" +
    '<div class="xqora-header-text">' +
    '<div class="xqora-title">XQORA BOT</div>' +
    '<div class="xqora-subtitle">Powered by AI</div>' +
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
  var inputBarEl = panel.querySelector("#xqora-chat-inputbar");

  // Hides/disables the input bar entirely (not just visually near the
  // "Start new chat" action) once a conversation has ended - re-shown by
  // startNewChat() when the fresh conversation begins.
  function setInputBarVisible(visible) {
    inputBarEl.style.display = visible ? "" : "none";
    inputEl.disabled = !visible;
    sendBtn.disabled = !visible || isSending;
  }

  function addQuickReplies() {
    var row = document.createElement("div");
    row.className = "xqora-quick-replies-row";
    QUICK_REPLIES.forEach(function (label) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "xqora-quick-reply-btn";
      btn.textContent = label;
      btn.addEventListener("click", function () {
        sendMessage(label);
      });
      row.appendChild(btn);
    });
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  var AUTO_RESTART_DELAY_MS = 2500;

  // Ends the current conversation in the UI: input bar is fully hidden (not
  // just a button placed near it) so it can't be used until a new chat
  // starts. When autoRestart is true (the post-feedback-submit path), a
  // fresh conversation starts on its own after a short delay so the user
  // doesn't have to click anything; the button still doubles as an
  // immediate override for anyone who doesn't want to wait.
  function addNewChatAction(autoRestart) {
    setInputBarVisible(false);

    var row = document.createElement("div");
    row.className = "xqora-quick-replies-row";

    var autoTimer = null;

    if (autoRestart) {
      var note = document.createElement("div");
      note.className = "xqora-auto-restart-note";
      note.textContent = "Starting a new chat...";
      row.appendChild(note);
    }

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "xqora-quick-reply-btn";
    btn.textContent = autoRestart ? "Start new chat now" : "Start new chat";
    btn.addEventListener("click", function () {
      if (autoTimer) clearTimeout(autoTimer);
      row.remove();
      startNewChat();
    });
    row.appendChild(btn);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    if (autoRestart) {
      autoTimer = setTimeout(function () {
        row.remove();
        startNewChat();
      }, AUTO_RESTART_DELAY_MS);
    }
  }

  // Resets the widget to a fresh conversation entirely in place - no
  // window.location.reload(), so the panel/open state, scroll position,
  // and rest of the host page are undisturbed. Clears this session's
  // rendered messages and its sessionStorage history/feedback-pending
  // entries, then rotates to a brand new session_id (same generation logic
  // as the initial-load one above) *before* rendering anything new, so the
  // fresh greeting/quick-replies get saved under the new session's history
  // key instead of the old one, and the next /chat request sent afterward
  // carries the new session_id - the backend (see app/routes/chat.py) just
  // takes whatever session_id it's given, so a new id there is genuinely a
  // new orchestrator session with no leftover lead/feedback state attached.
  function startNewChat() {
    removeFeedbackCard();
    _storageRemove(_historyKey());
    _storageRemove(_feedbackPendingKey());

    if (window.crypto && window.crypto.randomUUID) {
      sessionId = window.crypto.randomUUID();
    } else {
      sessionId = "xqora-" + Date.now() + "-" + Math.random().toString(16).slice(2);
    }
    _storageSet(SESSION_ID_STORAGE_KEY, sessionId);

    setInputBarVisible(true);
    messagesEl.innerHTML = "";
    addMessage(GREETING_TEXT, "bot");
    addQuickReplies();
    inputEl.focus();
  }

  function removeFeedbackCard() {
    if (activeFeedbackCard && activeFeedbackCard.parentNode) {
      activeFeedbackCard.remove();
    }
    activeFeedbackCard = null;
    _storageRemove(_feedbackPendingKey());
  }

  function stripFeedbackAsk(text) {
    var suffix = "\n\n" + FEEDBACK_ASK_TEXT;
    if (typeof text === "string" && text.slice(-suffix.length) === suffix) {
      return text.slice(0, -suffix.length);
    }
    return text;
  }

  function addFeedbackCard() {
    var row = document.createElement("div");
    row.className = "xqora-msg-row xqora-bot";

    var avatarEl = document.createElement("img");
    avatarEl.className = "xqora-bot-avatar";
    avatarEl.src = AVATAR_URL;
    avatarEl.alt = "";
    row.appendChild(avatarEl);

    var card = document.createElement("div");
    card.className = "xqora-feedback-card";

    var facesHtml = FEEDBACK_FACES.map(function (f) {
      return (
        '<button type="button" class="xqora-face-btn" data-rating="' + f.rating + '">' +
        '<span class="xqora-face-emoji">' + f.emoji + "</span>" +
        '<span class="xqora-face-label">' + f.label + "</span>" +
        "</button>"
      );
    }).join("");

    var reasonsHtml = FEEDBACK_REASONS.map(function (r) {
      return (
        '<button type="button" class="xqora-reason-chip" data-reason="' + r.reason + '">' +
        r.label +
        "</button>"
      );
    }).join("");

    card.innerHTML =
      '<div class="xqora-feedback-title">Thanks for chatting with us!</div>' +
      '<div class="xqora-feedback-subtitle">How would you rate your experience?</div>' +
      '<div class="xqora-feedback-faces">' + facesHtml + "</div>" +
      '<div class="xqora-feedback-reasons xqora-hidden">' + reasonsHtml + "</div>" +
      '<div class="xqora-feedback-actions">' +
      '<button type="button" class="xqora-feedback-cancel">Cancel</button>' +
      '<button type="button" class="xqora-feedback-submit" disabled>Submit</button>' +
      "</div>";

    row.appendChild(card);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    activeFeedbackCard = row;
    _storageSet(_feedbackPendingKey(), "1");

    var selectedRating = null;
    var selectedReason = null;
    var faceButtons = card.querySelectorAll(".xqora-face-btn");
    var reasonsRow = card.querySelector(".xqora-feedback-reasons");
    var reasonChips = card.querySelectorAll(".xqora-reason-chip");
    var cancelBtn = card.querySelector(".xqora-feedback-cancel");
    var submitBtn = card.querySelector(".xqora-feedback-submit");

    faceButtons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        selectedRating = btn.getAttribute("data-rating");
        faceButtons.forEach(function (b) {
          b.classList.toggle("xqora-selected", b === btn);
        });
        if (selectedRating === "happy") {
          reasonsRow.classList.add("xqora-hidden");
          selectedReason = null;
          reasonChips.forEach(function (c) {
            c.classList.remove("xqora-selected");
          });
        } else {
          reasonsRow.classList.remove("xqora-hidden");
        }
        submitBtn.disabled = false;
      });
    });

    reasonChips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        var alreadySelected = chip.classList.contains("xqora-selected");
        reasonChips.forEach(function (c) {
          c.classList.remove("xqora-selected");
        });
        if (alreadySelected) {
          selectedReason = null;
        } else {
          chip.classList.add("xqora-selected");
          selectedReason = chip.getAttribute("data-reason");
        }
      });
    });

    cancelBtn.addEventListener("click", function () {
      removeFeedbackCard();
      addNewChatAction();
      fetch(FEEDBACK_CANCEL_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }).catch(function () {});
    });

    submitBtn.addEventListener("click", function () {
      if (!selectedRating || submitBtn.disabled) return;
      submitBtn.disabled = true;
      cancelBtn.disabled = true;

      fetch(FEEDBACK_SUBMIT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          rating: selectedRating,
          reason: selectedReason,
          comment: null,
        }),
      })
        .then(function (res) {
          return res.json().then(function (data) {
            return { ok: res.ok, data: data };
          });
        })
        .then(function (result) {
          removeFeedbackCard();
          if (result.ok && result.data && result.data.reply) {
            addMessage(result.data.reply, "bot");
          }
          addNewChatAction(true);
        })
        .catch(function () {
          removeFeedbackCard();
          addMessage("Thanks for the feedback!", "bot");
          addNewChatAction(true);
        });
    });
  }

  function _renderMessageRow(text, sender) {
    var row = document.createElement("div");
    row.className = "xqora-msg-row " + (sender === "user" ? "xqora-user" : "xqora-bot");
    if (sender !== "user") {
      var avatarEl = document.createElement("img");
      avatarEl.className = "xqora-bot-avatar";
      avatarEl.src = AVATAR_URL;
      avatarEl.alt = "";
      row.appendChild(avatarEl);
    }
    var bubbleEl = document.createElement("div");
    bubbleEl.className = "xqora-bubble " + (sender === "user" ? "xqora-user" : "xqora-bot");
    bubbleEl.textContent = text;
    row.appendChild(bubbleEl);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return row;
  }

  function addMessage(text, sender) {
    var row = _renderMessageRow(text, sender);
    _appendHistoryEntry(sender, text);
    return row;
  }

  function addTypingIndicator() {
    var row = document.createElement("div");
    row.className = "xqora-msg-row xqora-bot";
    row.id = "xqora-typing-row";
    var avatarEl = document.createElement("img");
    avatarEl.className = "xqora-bot-avatar";
    avatarEl.src = AVATAR_URL;
    avatarEl.alt = "";
    row.appendChild(avatarEl);
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
    bubble.innerHTML = isOpen ? CLOSE_ICON_SVG : CHAT_ICON_IMG_HTML;
    bubble.classList.toggle("xqora-bubble-open", isOpen);
    bubble.setAttribute("aria-label", isOpen ? "Close XQORA chat" : "Open XQORA chat");
  }

  function openPanel() {
    isOpen = true;
    panel.classList.add("xqora-open");
    updateBubbleIcon();
    if (messagesEl.children.length === 0) {
      addMessage(GREETING_TEXT, "bot");
      addQuickReplies();
    }
    inputEl.focus();
  }

  function closePanel() {
    isOpen = false;
    panel.classList.remove("xqora-open");
    updateBubbleIcon();
  }

  // Restore this session's chat on load, before the panel is ever opened -
  // openPanel()'s "messagesEl.children.length === 0" check then naturally
  // skips the default greeting/quick-replies once history has been
  // rendered back in, same as it already skips them on a second openPanel()
  // call within one fresh session.
  (function restoreHistory() {
    var history = _loadHistory();
    if (history.length === 0) return;

    var sentByUser = false;
    history.forEach(function (entry) {
      if (!entry || typeof entry.text !== "string") return;
      var sender = entry.sender === "user" ? "user" : "bot";
      if (sender === "user") sentByUser = true;
      _renderMessageRow(entry.text, sender);
    });

    // No message from the user yet (they opened the widget, saw the
    // greeting + buttons, then reloaded before touching either) - restore
    // the buttons too, same rule as the live "hide after first send" logic.
    if (!sentByUser) {
      addQuickReplies();
    }

    if (_storageGet(_feedbackPendingKey()) === "1") {
      addFeedbackCard();
    }
  })();

  bubble.addEventListener("click", function () {
    isOpen ? closePanel() : openPanel();
  });

  function sendMessage(presetText) {
    var text = typeof presetText === "string" ? presetText.trim() : inputEl.value.trim();
    if (!text || isSending) return;

    var quickRepliesRow = messagesEl.querySelector(".xqora-quick-replies-row");
    if (quickRepliesRow) quickRepliesRow.remove();

    removeFeedbackCard();
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
          if (result.data.session_id && result.data.session_id !== sessionId) {
            // Server assigned a different session_id than the one we sent
            // (shouldn't normally happen now that we persist and resend our
            // own) - migrate this session's stored history/state to the new
            // key instead of leaving it orphaned under the stale one.
            var oldHistoryKey = _historyKey();
            var oldFeedbackKey = _feedbackPendingKey();
            var carriedHistory = _loadHistory();
            sessionId = result.data.session_id;
            _storageSet(SESSION_ID_STORAGE_KEY, sessionId);
            _storageSet(_historyKey(), JSON.stringify(carriedHistory));
            _storageRemove(oldHistoryKey);
            _storageRemove(oldFeedbackKey);
          }
          if (result.data.awaiting_feedback) {
            var cleanedReply = stripFeedbackAsk(result.data.reply);
            if (cleanedReply) addMessage(cleanedReply, "bot");
            addFeedbackCard();
          } else {
            addMessage(result.data.reply, "bot");
            // Quick replies are tied to "the greeting is the latest bot
            // reply", not "this is the very first message ever" - so they
            // reappear any time the user re-triggers the greeting (e.g.
            // typing "hi" again mid-conversation), not just on session start.
            if (result.data.is_greeting) addQuickReplies();
          }
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
