// Table viewer page.
//
// The <perspective-viewer> comes from the anywidget bundle shipped with
// perspective-python (so it always matches the server version). That bundle
// exposes `render({model, el})`, where `model` is a Jupyter widget model; we
// provide a small stand-in that forwards the widget's binary messages to the
// Perspective server websocket, and stores the viewer configuration (which the
// widget writes into the model) as the table's saved view.
import widget from "./perspective.js";

const POLL_MS = 2000;
const SAVE_DELAY_MS = 1000;

class SocketModel {
  constructor(tableName, attrs, onSave) {
    this.tableName = tableName;
    this.attrs = { ...attrs };
    this.onSave = onSave;
    this.listeners = new Map();
    this.clientId = null;
    this.queue = [];
    const url = new URL("websocket", window.location.href);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    this.ws = new WebSocket(url);
    this.ws.binaryType = "arraybuffer";
    this.ws.onopen = () => {
      for (const buffer of this.queue) this.ws.send(buffer);
      this.queue = [];
    };
    this.ws.onmessage = (event) => {
      const message = { type: "binary_msg", client_id: this.clientId };
      for (const listener of this.listeners.get("msg:custom") || []) {
        listener(message, [new DataView(event.data)]);
      }
    };
  }

  get(key) {
    if (key === "table_name") return this.tableName;
    if (key === "binding_mode") return "server";
    return this.attrs[key];
  }

  set(key, value) {
    this.attrs[key] = value;
  }

  save_changes() {
    this.onSave({ ...this.attrs });
  }

  on(event, listener) {
    if (!this.listeners.has(event)) this.listeners.set(event, []);
    this.listeners.get(event).push(listener);
  }

  off(event, listener) {
    const listeners = this.listeners.get(event) || [];
    this.listeners.set(event, listeners.filter((l) => l !== listener));
  }

  send(message, _callbacks, buffers) {
    if (message.type === "connect") {
      this.clientId = message.client_id;
    } else if (message.type === "binary_msg") {
      if (this.ws.readyState === WebSocket.OPEN) this.ws.send(buffers[0]);
      else this.queue.push(buffers[0]);
    } else if (message.type === "hangup") {
      this.ws.close();
    }
  }

  close() {
    this.ws.close();
  }
}

const state = {
  tables: [],
  selected: null, // logical table name
  mounted: null, // { name, hosted, model, cleanup }
};

function defaultTheme() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "Pro Dark"
    : "Pro Light";
}

// Perspective shows floats with 2 decimals, which hides small values such as
// learning rates (1e-4 -> 0.00): keep at least 4 significant digits instead.
const FLOAT_FORMAT = {
  maximumFractionDigits: 2,
  maximumSignificantDigits: 4,
  roundingPriority: "morePrecision",
};

function withNumberFormats(columnsConfig, types) {
  const config = { ...(columnsConfig || {}) };
  for (const [column, type] of Object.entries(types || {})) {
    if (type !== "float") continue;
    const current = config[column] || {};
    if (!current.number_format) {
      config[column] = { ...current, number_format: { ...FLOAT_FORMAT } };
    }
  }
  return config;
}

// A saved view lists its visible columns, so columns that appeared since (e.g. a
// task wrote a new metric) would stay hidden: show them in the grid view, and
// remember which columns the view has seen (`known_columns`, our own key).
function showNewColumns(attrs, columns) {
  const known = attrs.known_columns;
  attrs.known_columns = columns;
  if (!known || !Array.isArray(attrs.columns)) return;
  if (attrs.plugin && attrs.plugin !== "Datagrid") return;
  const added = columns.filter((c) => !known.includes(c) && !attrs.columns.includes(c));
  attrs.columns = [...attrs.columns, ...added];
}

function debounce(fn, delay) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

async function loadView(name) {
  try {
    const response = await fetch(`api/views/${encodeURIComponent(name)}`);
    return (await response.json()) || {};
  } catch (e) {
    console.warn("Could not load the saved view", e);
    return {};
  }
}

function saver(name) {
  return debounce(async (attrs) => {
    // The theme follows the browser's light/dark preference
    const { theme: _theme, ...view } = attrs;
    try {
      await fetch(`api/views/${encodeURIComponent(name)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(view),
      });
    } catch (e) {
      console.warn("Could not save the view", e);
    }
  }, SAVE_DELAY_MS);
}

async function unmount() {
  const mounted = state.mounted;
  state.mounted = null;
  if (!mounted) return mounted;
  try {
    if (mounted.cleanup) await mounted.cleanup();
  } finally {
    mounted.model.close();
  }
  return mounted;
}

async function mount(table) {
  const container = document.getElementById("viewer");
  const previous = await unmount();
  container.replaceChildren();
  if (!table.table) {
    container.innerHTML = '<div class="empty">No rows loaded yet.</div>';
    return;
  }

  // Remounting the same table (new columns): keep the current configuration
  const attrs =
    previous && previous.name === table.name
      ? previous.model.attrs
      : await loadView(table.name);
  if (!attrs.theme) attrs.theme = defaultTheme();
  if (!attrs.title) attrs.title = table.name;
  attrs.columns_config = withNumberFormats(attrs.columns_config, table.types);
  showNewColumns(attrs, Object.keys(table.types || {}));

  const model = new SocketModel(table.table, attrs, saver(table.name));
  const mounted = { name: table.name, hosted: table.table, model, cleanup: null };
  state.mounted = mounted;
  mounted.cleanup = await widget.render({ model, el: container });
}

function renderList() {
  const list = document.getElementById("tables");
  if (!state.tables.length) {
    list.innerHTML = '<div class="empty">No table registered yet.</div>';
    return;
  }
  list.replaceChildren(
    ...state.tables.map((table) => {
      const item = document.createElement("div");
      item.className = "table" + (table.name === state.selected ? " selected" : "");
      const counts = {};
      for (const file of table.files) counts[file.status] = (counts[file.status] || 0) + 1;
      const meta = [`${table.rows} rows`, `${table.files.length} file(s)`];
      if (counts.pending) meta.push(`${counts.pending} pending`);
      if (counts.error) meta.push(`${counts.error} error(s)`);

      const name = document.createElement("div");
      name.className = "name";
      name.textContent = table.name;
      const info = document.createElement("div");
      info.className = "meta";
      info.textContent = meta.join(" · ");
      item.append(name, info);

      if (table.name === state.selected) {
        const files = document.createElement("ul");
        files.className = "files";
        for (const file of table.files) {
          const li = document.createElement("li");
          li.className = file.status;
          li.textContent = `${file.run || file.path.split("/").pop()} — ${file.status}`;
          li.title = file.error ? `${file.path}\n${file.error}` : file.path;
          files.append(li);
        }
        item.append(files);
      }
      item.onclick = () => select(table.name);
      return item;
    }),
  );
}

async function select(name) {
  state.selected = name;
  const url = new URL(window.location.href);
  url.hash = encodeURIComponent(name);
  history.replaceState(null, "", url);
  renderList();
  const table = state.tables.find((t) => t.name === name);
  if (table) await mount(table);
}

async function poll() {
  try {
    const response = await fetch("api/tables");
    state.tables = (await response.json()).tables;
  } catch (e) {
    console.warn("Could not list the tables", e);
    return;
  }

  if (state.selected === null) {
    const fromHash = decodeURIComponent(window.location.hash.slice(1));
    const initial = state.tables.find((t) => t.name === fromHash) || state.tables[0];
    if (initial) return select(initial.name);
  }
  renderList();

  // The selected table got rows, or a new generation (its columns changed)
  const table = state.tables.find((t) => t.name === state.selected);
  const hosted = state.mounted ? state.mounted.hosted : null;
  if (table && table.table !== hosted) await mount(table);
}

async function loop() {
  await poll();
  setTimeout(loop, POLL_MS);
}

loop();
