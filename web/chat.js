const form = document.querySelector("form");
const question = document.querySelector("#question");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const send = document.querySelector("#send");
const reset = document.querySelector("#reset");

function addMessage(author, content, className = "") {
  const article = document.createElement("article");
  article.className = className;
  const label = document.createElement("strong");
  label.textContent = author;
  const text = document.createElement("p");
  text.textContent = content;
  article.append(label, text);
  messages.append(article);
  return { article, text };
}

function addSources(article, sources, title) {
  const list = document.createElement("ul");
  for (const { document: filename, page, source_id: id } of sources) {
    if (!filename) continue;
    const item = document.createElement("li");
    const name = filename.split(/[\\/]/).pop();
    const reference = page == null ? name : name + " · página " + page;
    item.textContent = id == null ? reference : "Fuente " + id + " · " + reference;
    list.append(item);
  }
  if (!list.childElementCount) return;
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = title;
  details.append(summary, list);
  article.append(details);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = question.value.trim();
  if (!prompt || send.disabled) return;

  welcome.hidden = true;
  reset.hidden = false;
  reset.disabled = true;
  send.disabled = true;
  question.value = "";
  addMessage("Tú", prompt, "user");
  const reply = addMessage("Asistente Unison", "Consultando documentos…");
  reply.article.scrollIntoView({ block: "start" });

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: prompt }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    reply.text.textContent = data.answer;
    if (data.citations?.length) {
      addSources(reply.article, data.citations, "Fuentes citadas");
    } else {
      addSources(reply.article, data.sources, "Fuentes recuperadas");
    }
  } catch (error) {
    reply.text.textContent = error instanceof TypeError
      ? "Se perdió la conexión con el servidor local. Inténtalo de nuevo."
      : error.message;
    if (!question.value) question.value = prompt;
  } finally {
    send.disabled = false;
    reset.disabled = false;
  }
});

document.querySelector("#suggestions").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  question.value = button.textContent;
  form.requestSubmit();
});

reset.addEventListener("click", () => {
  messages.replaceChildren();
  question.value = "";
  welcome.hidden = false;
  reset.hidden = true;
  question.focus();
});
