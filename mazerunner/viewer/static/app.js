// ─── State ──────────────────────────────────────────────────────

let currentMode = "replay";  // "replay" or "live"
let currentEval = null;
let currentEpisode = null;
let currentStepIndex = -1;   // -1 = initial observation
let autoplayTimer = null;
let liveWs = null;
let liveSteps = [];
let liveInitial = null;

// ─── Mode Switching ─────────────────────────────────────────────

function switchMode(mode) {
    currentMode = mode;
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.mode === mode));
    document.getElementById("replay-controls").style.display = mode === "replay" ? "flex" : "none";
    document.getElementById("live-controls").style.display = mode === "live" ? "flex" : "none";

    if (mode === "live") {
        loadInstances();
    }

    clearTrajectory();
}

// ─── Replay Mode ────────────────────────────────────────────────

async function loadEvalList() {
    const res = await fetch("/api/evals");
    const evals = await res.json();
    const sel = document.getElementById("eval-select");
    sel.innerHTML = '<option value="">Select eval file...</option>';
    evals.forEach(e => {
        const opt = document.createElement("option");
        opt.value = e.filename;
        opt.textContent = `${e.filename} (${e.model}, ${e.mode}, ${e.num_episodes} ep)`;
        sel.appendChild(opt);
    });
}

async function loadEval(filename) {
    if (!filename) return;
    const res = await fetch(`/api/evals/${filename}`);
    currentEval = await res.json();

    const sel = document.getElementById("episode-select");
    sel.innerHTML = '<option value="">Select episode...</option>';
    sel.disabled = false;

    currentEval.records.forEach((rec, i) => {
        const opt = document.createElement("option");
        opt.value = i;
        const status = rec.success ? "SUCCESS" : "FAIL";
        opt.textContent = `${rec.maze_id} - ${status} (${rec.steps} steps)`;
        sel.appendChild(opt);
    });

    clearTrajectory();
}

async function loadEpisode(index) {
    if (index === "" || !currentEval) return;
    currentEpisode = currentEval.records[parseInt(index)];
    currentStepIndex = -1;
    renderTrajectory();
    selectStep(-1);
}

// ─── Trajectory Rendering ───────────────────────────────────────

function clearTrajectory() {
    currentEpisode = null;
    currentStepIndex = -1;
    liveSteps = [];
    liveInitial = null;
    stopAutoplay();
    document.getElementById("trajectory-list").innerHTML = "";
    document.getElementById("state-display").innerHTML = '<p class="placeholder">Select an eval file and episode to view</p>';
    document.getElementById("step-counter").textContent = "-";
    document.getElementById("episode-meta").textContent = "";
    document.getElementById("btn-prev").disabled = true;
    document.getElementById("btn-next").disabled = true;
}

function renderTrajectory() {
    const list = document.getElementById("trajectory-list");
    list.innerHTML = "";

    const episode = currentEpisode;
    if (!episode) return;

    const mode = episode.mode || currentEval?.mode || "text_grid";
    const meta = document.getElementById("episode-meta");
    const status = episode.success ? "SUCCESS" : "FAIL";
    meta.textContent = `${episode.maze_id} | ${status} | ${episode.steps} steps | reward: ${episode.reward.toFixed(2)}`;

    // Initial observation card
    if (episode.initial_observation && episode.initial_observation.rendered) {
        const card = createStepCard(-1, {
            label: "Initial State",
            tool: `mode: ${mode} | position: [${(episode.initial_observation.position || []).join(", ")}]`,
            valid: true,
            finished: false,
            reward: 0,
            reasoning: "",
            isInitial: true,
        });
        list.appendChild(card);
    }

    // Step cards
    episode.trajectory.forEach((step, i) => {
        const args = JSON.stringify(step.action);
        const card = createStepCard(i, {
            label: `Step ${i}`,
            tool: `${step.tool_name}(${args})`,
            valid: step.valid,
            finished: step.raw_result?.finished || false,
            reward: step.reward,
            reasoning: step.reasoning || "",
            isInitial: false,
        });
        list.appendChild(card);
    });

    updateStepControls();
}

function createStepCard(index, data) {
    const card = document.createElement("div");
    let cls = "step-card";
    if (data.isInitial) cls += " initial";
    else if (data.finished) cls += " finished";
    else if (data.valid) cls += " valid";
    else cls += " invalid";
    if (index === currentStepIndex) cls += " active";
    card.className = cls;
    card.onclick = () => selectStep(index);

    let statusHtml = "";
    if (!data.isInitial) {
        statusHtml = `<div class="step-status">`;
        statusHtml += data.valid
            ? `<span class="valid-badge">&#10003; Valid</span>`
            : `<span class="invalid-badge">&#10007; Invalid</span>`;
        statusHtml += `<span class="reward-badge">reward: ${data.reward.toFixed(2)}</span>`;
        if (data.finished) statusHtml += `<span class="finished-badge">GOAL!</span>`;
        statusHtml += `</div>`;
    }

    let reasoningHtml = "";
    if (data.reasoning) {
        const truncated = data.reasoning.length > 500
            ? data.reasoning.slice(0, 500) + "..."
            : data.reasoning;
        reasoningHtml = `<div class="step-reasoning">${escapeHtml(truncated)}</div>`;
    }

    card.innerHTML = `
        <div class="step-label">${data.label}</div>
        <div class="step-tool">${escapeHtml(data.tool)}</div>
        ${statusHtml}
        ${reasoningHtml}
    `;
    return card;
}

function selectStep(index) {
    currentStepIndex = index;

    // Update active card
    document.querySelectorAll(".step-card").forEach((card, i) => {
        // Cards are indexed: -1 (initial) maps to DOM index 0, step 0 to DOM index 1, etc.
        const cardIndex = i - (currentEpisode?.initial_observation?.rendered ? 1 : 0);
        card.classList.toggle("active", cardIndex === index);
    });

    // Update maze state display
    const episode = currentEpisode;
    if (!episode) return;

    const mode = episode.mode || currentEval?.mode || "text_grid";
    let rendered = null;

    if (index === -1 && episode.initial_observation) {
        rendered = episode.initial_observation.rendered;
    } else if (index >= 0 && index < episode.trajectory.length) {
        rendered = episode.trajectory[index].raw_result?.rendered;
    }

    renderMazeState(rendered, mode);
    updateStepControls();

    // Scroll the active card into view
    const cards = document.querySelectorAll(".step-card");
    const activeIdx = index + (episode.initial_observation?.rendered ? 1 : 0);
    if (activeIdx >= 0 && activeIdx < cards.length) {
        cards[activeIdx].scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
}

function renderMazeState(rendered, mode) {
    const display = document.getElementById("state-display");

    if (!rendered) {
        display.innerHTML = '<p class="placeholder">No rendered state available</p>';
        return;
    }

    if (mode === "text_grid") {
        display.innerHTML = `<pre>${escapeHtml(rendered)}</pre>`;
    } else {
        // Vision modes: rendered is base64 PNG
        display.innerHTML = `<img src="data:image/png;base64,${rendered}" alt="Maze state">`;
    }
}

function updateStepControls() {
    const episode = currentEpisode;
    const hasInitial = episode?.initial_observation?.rendered;
    const minStep = hasInitial ? -1 : 0;
    const maxStep = episode ? episode.trajectory.length - 1 : -1;

    document.getElementById("btn-prev").disabled = !episode || currentStepIndex <= minStep;
    document.getElementById("btn-next").disabled = !episode || currentStepIndex >= maxStep;

    if (episode) {
        const total = episode.trajectory.length + (hasInitial ? 1 : 0);
        const current = currentStepIndex + (hasInitial ? 2 : 1);
        document.getElementById("step-counter").textContent = `${current} / ${total}`;
    }
}

function prevStep() {
    if (!currentEpisode) return;
    const minStep = currentEpisode.initial_observation?.rendered ? -1 : 0;
    if (currentStepIndex > minStep) selectStep(currentStepIndex - 1);
}

function nextStep() {
    if (!currentEpisode) return;
    if (currentStepIndex < currentEpisode.trajectory.length - 1) selectStep(currentStepIndex + 1);
}

function toggleAutoplay() {
    if (autoplayTimer) {
        stopAutoplay();
    } else {
        startAutoplay();
    }
}

function startAutoplay() {
    const btn = document.getElementById("btn-play");
    btn.textContent = "\u25A0 Stop";
    autoplayTimer = setInterval(() => {
        if (!currentEpisode || currentStepIndex >= currentEpisode.trajectory.length - 1) {
            stopAutoplay();
            return;
        }
        nextStep();
    }, 800);
}

function stopAutoplay() {
    if (autoplayTimer) {
        clearInterval(autoplayTimer);
        autoplayTimer = null;
    }
    document.getElementById("btn-play").textContent = "\u25B6 Play";
}

// ─── Live Mode ──────────────────────────────────────────────────

async function loadInstances() {
    const res = await fetch("/api/instances");
    const instances = await res.json();
    const sel = document.getElementById("live-instance");
    sel.innerHTML = '<option value="">Select maze...</option>';
    instances.forEach(name => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
    });
}

function startLive() {
    const provider = document.getElementById("live-provider").value;
    const model = document.getElementById("live-model").value;
    const mode = document.getElementById("live-mode").value;
    const instance = document.getElementById("live-instance").value;
    const singleStep = document.getElementById("live-single-step").checked;
    const maxTurns = parseInt(document.getElementById("live-max-turns").value);

    if (!instance) {
        alert("Please select a maze instance");
        return;
    }

    // Close existing connection
    if (liveWs) {
        liveWs.close();
        liveWs = null;
    }

    clearTrajectory();
    liveSteps = [];
    liveInitial = null;

    // Build a fake episode for rendering
    currentEpisode = {
        maze_id: instance,
        success: false,
        steps: 0,
        reward: 0,
        trajectory: [],
        mode: mode,
        initial_observation: {},
    };

    const btn = document.getElementById("live-start-btn");
    btn.textContent = "Running...";
    btn.disabled = true;

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    liveWs = new WebSocket(`${proto}//${location.host}/ws/live`);

    liveWs.onopen = () => {
        liveWs.send(JSON.stringify({
            provider, model, mode, instance, max_turns: maxTurns, single_step: singleStep,
        }));
    };

    liveWs.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleLiveMessage(msg);
    };

    liveWs.onerror = () => {
        btn.textContent = "Start";
        btn.disabled = false;
        document.getElementById("episode-meta").textContent = "WebSocket error";
    };

    liveWs.onclose = () => {
        btn.textContent = "Start";
        btn.disabled = false;
    };
}

function handleLiveMessage(msg) {
    switch (msg.type) {
        case "config":
            document.getElementById("episode-meta").textContent =
                `Live: ${msg.data.provider}/${msg.data.model} on ${msg.data.instance}`;
            break;

        case "initial":
            liveInitial = msg.data;
            currentEpisode.initial_observation = msg.data;
            currentEpisode.mode = msg.data.mode || currentEpisode.mode;
            renderTrajectory();
            selectStep(-1);
            break;

        case "step":
            const step = msg.data;
            liveSteps.push(step);
            currentEpisode.trajectory.push(step);
            currentEpisode.steps = liveSteps.length;
            currentEpisode.reward += step.reward;
            renderTrajectory();
            selectStep(liveSteps.length - 1);
            break;

        case "done":
            currentEpisode.success = msg.data.success;
            currentEpisode.reward = msg.data.total_reward;
            currentEpisode.steps = msg.data.total_steps;
            const status = msg.data.success ? "SUCCESS" : "FAIL";
            document.getElementById("episode-meta").textContent =
                `${msg.data.maze_id} | ${status} | ${msg.data.total_steps} steps | reward: ${msg.data.total_reward.toFixed(2)}`;
            break;

        case "error":
            document.getElementById("episode-meta").textContent = `Error: ${msg.data.message}`;
            break;
    }
}

// ─── Utilities ──────────────────────────────────────────────────

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); prevStep(); }
    if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); nextStep(); }
    if (e.key === " ") { e.preventDefault(); toggleAutoplay(); }
});

// ─── Init ───────────────────────────────────────────────────────

loadEvalList();
