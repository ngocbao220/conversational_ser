const state = {
  data: null,
  session: "Ses05",
  dialogue: "all",
  model: "tim",
  view: "full",
  pairedOutcome: "all",
  labels: new Set(),
  search: "",
  player: {
    playlist: [],
    index: 0,
    offset: 0,
    startedOffset: 0,
    startedAt: 0,
    duration: 0,
    baseTime: 0,
    playing: false,
    timers: [],
    activeAudios: [],
    progressTimer: null,
  },
};

const colorVars = {
  angry: "var(--angry)",
  happy: "var(--happy)",
  neutral: "var(--neutral)",
  sad: "var(--sad)",
};

const nodes = {
  utteranceCount: document.querySelector("#utteranceCount"),
  dialogueCount: document.querySelector("#dialogueCount"),
  sessionCount: document.querySelector("#sessionCount"),
  sessionFilters: document.querySelector("#sessionFilters"),
  dialogueSelect: document.querySelector("#dialogueSelect"),
  viewControls: document.querySelector("#viewControls"),
  pairControls: document.querySelector("#pairControls"),
  modelControls: document.querySelector("#modelControls"),
  labelFilters: document.querySelector("#labelFilters"),
  activeDialogue: document.querySelector("#activeDialogue"),
  streamMeta: document.querySelector("#streamMeta"),
  searchInput: document.querySelector("#searchInput"),
  playDialogueButton: document.querySelector("#playDialogueButton"),
  prevTurnButton: document.querySelector("#prevTurnButton"),
  nextTurnButton: document.querySelector("#nextTurnButton"),
  stopDialogueButton: document.querySelector("#stopDialogueButton"),
  playerTitle: document.querySelector("#playerTitle"),
  playerSubtitle: document.querySelector("#playerSubtitle"),
  playerProgress: document.querySelector("#playerProgress"),
  labelBars: document.querySelector("#labelBars"),
  evidencePanel: document.querySelector("#evidencePanel"),
  streamList: document.querySelector("#streamList"),
};

const views = [
  { id: "full", name: "Full dialogue", help: "All Session 5 turns, preserving context" },
  { id: "evaluated", name: "Evaluated only", help: "Only utterances with model predictions" },
  { id: "tim_fixes", name: "TIM fixes only", help: "TIM correct while baseline or MAL is wrong" },
];

const pairedOutcomes = [
  { id: "all", name: "All paired outcomes", help: "Keep every turn in the current view" },
  { id: "tim_only_correct", name: "TIM correct, MAL wrong", help: "TIM-only wins" },
  { id: "mal_only_correct", name: "MAL correct, TIM wrong", help: "MAL-only wins" },
  { id: "both_correct", name: "Both correct", help: "MAL and TIM match the gold label" },
  { id: "both_wrong", name: "Both wrong", help: "Both models miss the gold label" },
];

fetch(`demo_data.json?v=${Date.now()}`, { cache: "no-store" })
  .then((response) => response.json())
  .then((data) => {
    state.data = data;
    nodes.utteranceCount.textContent = data.summary.utterance_count.toLocaleString();
    nodes.dialogueCount.textContent = data.summary.dialogue_count.toLocaleString();
    nodes.sessionCount.textContent = data.summary.session_count.toLocaleString();
    renderControls();
    render();
  })
  .catch((error) => {
    nodes.streamList.innerHTML = `<div class="empty-state">Cannot load demo_data.json: ${escapeHtml(error.message)}</div>`;
  });

nodes.searchInput.addEventListener("input", (event) => {
  state.search = event.target.value.trim().toLowerCase();
  render();
});

nodes.dialogueSelect.addEventListener("change", (event) => {
  state.dialogue = event.target.value;
  stopDialogue();
  render();
});

nodes.playDialogueButton.addEventListener("click", () => {
  if (state.player.playing) {
    pauseDialogue();
  } else {
    playDialogue();
  }
});

nodes.stopDialogueButton.addEventListener("click", () => {
  stopDialogue();
});

nodes.prevTurnButton.addEventListener("click", () => {
  stepDialogue(-1);
});

nodes.nextTurnButton.addEventListener("click", () => {
  stepDialogue(1);
});

function renderControls() {
  renderSessionFilters();
  renderDialogueOptions();
  renderViewControls();
  renderPairControls();
  renderModelControls();
  renderLabelFilters();
}

function renderSessionFilters() {
  const sessions = [{ id: "all", count: state.data.summary.utterance_count }, ...state.data.sessions];
  nodes.sessionFilters.innerHTML = sessions
    .map((session) => {
      const label = session.id === "all" ? "All" : session.id;
      const active = state.session === session.id ? " active" : "";
      return `<button class="chip${active}" data-session="${session.id}">${label} <small>${session.count}</small></button>`;
    })
    .join("");

  nodes.sessionFilters.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.session = button.dataset.session;
      state.dialogue = "all";
      stopDialogue();
      renderSessionFilters();
      renderDialogueOptions();
      render();
    });
  });
}

function renderDialogueOptions() {
  const dialogues = getDialoguesForSession();
  nodes.dialogueSelect.innerHTML = [
    `<option value="all">All dialogues (${countForDialogues(dialogues)} turns)</option>`,
    ...dialogues.map((dialogue) => {
      const predicted = dialogue.predicted_count || 0;
      return `<option value="${dialogue.id}">${dialogue.id} (${dialogue.count} turns · ${predicted} predicted)</option>`;
    }),
  ].join("");
  nodes.dialogueSelect.value = state.dialogue;
}

function renderViewControls() {
  nodes.viewControls.innerHTML = views
    .map((view) => {
      const active = state.view === view.id ? " active" : "";
      return `<button class="segment${active}" data-view="${view.id}"><span>${view.name}</span><small>${view.help}</small></button>`;
    })
    .join("");

  nodes.viewControls.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.view = button.dataset.view;
      renderViewControls();
      render();
    });
  });
}

function renderPairControls() {
  const counts = state.data.summary.mal_tim_paired_counts || {};
  nodes.pairControls.innerHTML = pairedOutcomes
    .map((outcome) => {
      const active = state.pairedOutcome === outcome.id ? " active" : "";
      const count = outcome.id === "all"
        ? state.data.summary.fully_compared_count
        : (counts[outcome.id] || 0);
      return `<button class="segment${active}" data-paired-outcome="${outcome.id}"><span>${outcome.name} <em>${Number(count).toLocaleString()}</em></span><small>${outcome.help}</small></button>`;
    })
    .join("");

  nodes.pairControls.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.pairedOutcome = button.dataset.pairedOutcome;
      renderPairControls();
      render();
    });
  });
}

function renderModelControls() {
  nodes.modelControls.innerHTML = state.data.models
    .map((model) => {
      const active = state.model === model.id ? " active" : "";
      return `<button class="segment${active}" data-model="${model.id}">${model.name}</button>`;
    })
    .join("");

  nodes.modelControls.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.model = button.dataset.model;
      renderModelControls();
      render();
    });
  });
}

function renderLabelFilters() {
  nodes.labelFilters.innerHTML = state.data.labels
    .map((label) => {
      const active = state.labels.has(label) ? " active" : "";
      return `<button class="chip${active}" data-label="${label}">${label}</button>`;
    })
    .join("");

  nodes.labelFilters.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      const label = button.dataset.label;
      if (state.labels.has(label)) {
        state.labels.delete(label);
      } else {
        state.labels.add(label);
      }
      renderLabelFilters();
      render();
    });
  });
}

function render() {
  const utterances = filteredUtterances();
  const activeModel = state.data.models.find((model) => model.id === state.model);
  const predictedCount = utterances.filter((item) => item.predictions[state.model]).length;
  nodes.activeDialogue.textContent = state.dialogue === "all" ? "All dialogues" : state.dialogue;
  nodes.streamMeta.textContent = `${utterances.length.toLocaleString()} turns shown · ${predictedCount.toLocaleString()} have ${activeModel.name} predictions`;
  renderLabelBars(utterances);
  renderEvidencePanel(utterances);
  renderStream(utterances);
  syncPlayerPlaylist();
  renderPlayer();
}

function filteredUtterances() {
  return state.data.utterances.filter((item) => {
    const prediction = item.predictions[state.model];
    if (state.session !== "all" && item.session_id !== state.session) return false;
    if (state.dialogue !== "all" && item.dialogue_id !== state.dialogue) return false;
    if (state.view === "evaluated" && !prediction) return false;
    if (state.view === "tim_fixes" && !isTimFix(item)) return false;
    if (state.pairedOutcome !== "all" && item.comparison?.mal_tim_outcome !== state.pairedOutcome) return false;
    if (state.labels.size > 0 && (!prediction || !state.labels.has(prediction.label))) return false;
    if (!state.search) return true;
    const haystack = [
      item.utterance_id,
      item.dialogue_id,
      item.speaker_id,
      item.transcript,
      item.gold_label,
      prediction?.label || "",
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(state.search);
  });
}

function isTimFix(item) {
  const comparison = item.comparison || {};
  return comparison.outcome === "tim_correct_baseline_mal_wrong"
    || comparison.outcome === "tim_correct_baseline_wrong"
    || comparison.outcome === "tim_correct_mal_wrong";
}

function renderLabelBars(utterances) {
  const counts = Object.fromEntries(state.data.labels.map((label) => [label, 0]));
  utterances.forEach((item) => {
    const label = item.predictions[state.model]?.label;
    if (label in counts) counts[label] += 1;
  });
  const max = Math.max(1, ...Object.values(counts));

  nodes.labelBars.innerHTML = state.data.labels
    .map((label) => {
      const width = Math.round((counts[label] / max) * 100);
      return `
        <div class="label-bar">
          <header><span>${label}</span><span>${counts[label]}</span></header>
          <div class="bar-track"><div class="bar-fill" style="width: ${width}%; background: ${colorVars[label]}"></div></div>
        </div>
      `;
    })
    .join("");
}

function renderEvidencePanel(utterances) {
  const compared = utterances.filter((item) => item.comparison?.has_all_predictions);
  const timCorrect = compared.filter((item) => item.comparison.tim_correct).length;
  const baselineCorrect = compared.filter((item) => item.comparison.baseline_correct).length;
  const malCorrect = compared.filter((item) => item.comparison.mal_correct).length;
  const timFixBaseline = compared.filter((item) => item.comparison.outcome === "tim_correct_baseline_wrong" || item.comparison.outcome === "tim_correct_baseline_mal_wrong").length;
  const timFixMal = compared.filter((item) => item.comparison.outcome === "tim_correct_mal_wrong" || item.comparison.outcome === "tim_correct_baseline_mal_wrong").length;
  const timFixBoth = compared.filter((item) => item.comparison.outcome === "tim_correct_baseline_mal_wrong").length;

  nodes.evidencePanel.innerHTML = [
    renderEvidenceMetric("Compared", compared.length, "utterances with all 3 predictions"),
    renderEvidenceMetric("TIM correct", timCorrect, percent(timCorrect, compared.length)),
    renderEvidenceMetric("Baseline correct", baselineCorrect, percent(baselineCorrect, compared.length)),
    renderEvidenceMetric("MAL correct", malCorrect, percent(malCorrect, compared.length)),
    renderEvidenceMetric("TIM fixes baseline", timFixBaseline, "baseline wrong, TIM correct"),
    renderEvidenceMetric("TIM fixes MAL", timFixMal, "MAL wrong, TIM correct"),
    renderEvidenceMetric("TIM fixes both", timFixBoth, "baseline and MAL wrong"),
  ].join("");
}

function renderEvidenceMetric(label, value, note) {
  return `
    <div class="evidence-metric">
      <span>${label}</span>
      <strong>${Number(value).toLocaleString()}</strong>
      <small>${note}</small>
    </div>
  `;
}

function renderStream(utterances) {
  if (!utterances.length) {
    nodes.streamList.innerHTML = '<div class="empty-state">No utterances match the current filters.</div>';
    return;
  }

  nodes.streamList.innerHTML = utterances.map(renderUtterance).join("");
}

function renderUtterance(item) {
  const prediction = item.predictions[state.model];
  const audio = item.audio_path
    ? `<audio class="audio" controls preload="none" src="../${escapeAttribute(item.audio_path)}"></audio>`
    : "";
  const labelClass = prediction?.label || "unpredicted";
  const proofClass = isTimFix(item) ? " tim-proof" : "";

  return `
    <article class="utterance ${labelClass}${proofClass}" id="turn-${escapeAttribute(item.utterance_id)}">
      <div class="speaker">
        <strong>${escapeHtml(item.speaker_id || "speaker")}</strong>
        <span>${escapeHtml(item.utterance_id)}</span>
        <span>${formatTurnMeta(item)}</span>
      </div>
      <div class="transcript">
        <p>${escapeHtml(item.transcript || "(no transcript)")}</p>
        ${audio}
      </div>
      ${prediction ? renderPrediction(item, prediction) : renderMissingPrediction(item)}
    </article>
  `;
}

function syncPlayerPlaylist() {
  const playlist = dialoguePlaylist();
  const current = state.player.playlist[state.player.index];
  state.player.playlist = playlist;
  state.player.baseTime = playlist.length ? Number(playlist[0].start_time || 0) : 0;
  state.player.duration = playlist.length
    ? Math.max(...playlist.map((item) => Number(item.end_time || item.start_time || 0))) - state.player.baseTime
    : 0;

  if (!playlist.length) {
    state.player.index = 0;
    state.player.offset = 0;
    return;
  }

  if (current) {
    const nextIndex = playlist.findIndex((item) => item.utterance_id === current.utterance_id);
    state.player.index = nextIndex >= 0 ? nextIndex : 0;
  } else {
    state.player.index = 0;
  }
}

function dialoguePlaylist() {
  if (state.dialogue === "all") return [];
  return state.data.utterances
    .filter((item) => item.dialogue_id === state.dialogue && item.audio_path && item.start_time !== null && item.end_time !== null)
    .sort((a, b) => {
      const timeDelta = Number(a.start_time) - Number(b.start_time);
      if (timeDelta !== 0) return timeDelta;
      const turnA = Number.isInteger(a.turn_index) ? a.turn_index : 10_000;
      const turnB = Number.isInteger(b.turn_index) ? b.turn_index : 10_000;
      return turnA - turnB;
    });
}

function playDialogue() {
  syncPlayerPlaylist();
  if (!state.player.playlist.length) {
    renderPlayer("Select one dialogue before playback.");
    return;
  }

  startTimeline(state.player.offset);
}

function pauseDialogue() {
  if (state.player.playing) {
    state.player.offset = currentOffset();
  }
  clearTimelinePlayback();
  state.player.playing = false;
  renderPlayer();
}

function stopDialogue() {
  clearTimelinePlayback();
  state.player.index = 0;
  state.player.offset = 0;
  state.player.playing = false;
  renderPlayer();
  clearActiveTurn();
}

function stepDialogue(step) {
  syncPlayerPlaylist();
  if (!state.player.playlist.length) {
    renderPlayer("Select one dialogue before playback.");
    return;
  }
  const currentIndex = currentTimelineIndex();
  const nextIndex = clamp(currentIndex + step, 0, state.player.playlist.length - 1);
  jumpToIndex(nextIndex);
}

function startTimeline(offset) {
  const playlist = state.player.playlist;
  if (!playlist.length) {
    state.player.playing = false;
    renderPlayer("Select one dialogue before playback.");
    return;
  }

  clearTimelinePlayback();
  state.player.offset = clamp(offset, 0, Math.max(0, state.player.duration));
  if (state.player.offset >= state.player.duration) {
    state.player.offset = 0;
  }
  state.player.startedOffset = state.player.offset;
  state.player.startedAt = performance.now();
  state.player.playing = true;
  scheduleTimeline();
  state.player.progressTimer = window.setInterval(() => {
    const offsetNow = currentOffset();
    state.player.offset = offsetNow;
    state.player.index = currentTimelineIndex(offsetNow);
    renderPlayer();
    if (offsetNow >= state.player.duration) {
      finishTimeline();
    }
  }, 250);
  renderPlayer();
}

function scheduleTimeline() {
  const offset = state.player.offset;
  state.player.playlist.forEach((item) => {
    const startOffset = Number(item.start_time) - state.player.baseTime;
    const endOffset = Number(item.end_time) - state.player.baseTime;
    if (endOffset <= offset) return;

    const playItem = () => {
      const audioOffset = Math.max(0, currentOffset() - startOffset);
      playTimelineAudio(item, audioOffset);
      state.player.index = state.player.playlist.findIndex((row) => row.utterance_id === item.utterance_id);
      highlightActiveTurn(item.utterance_id);
      renderPlayer();
    };

    if (startOffset <= offset) {
      playItem();
      return;
    }

    const timer = window.setTimeout(playItem, (startOffset - offset) * 1000);
    state.player.timers.push(timer);
  });

  const finishTimer = window.setTimeout(finishTimeline, Math.max(0, state.player.duration - offset) * 1000 + 250);
  state.player.timers.push(finishTimer);
}

function playTimelineAudio(item, audioOffset) {
  const audio = new Audio(`../${item.audio_path}`);
  audio.preload = "auto";
  audio.currentTime = audioOffset;
  audio.play().catch(() => {
    renderPlayer("Browser blocked one clip. Press Play again if audio stops.");
  });
  state.player.activeAudios.push(audio);
}

function jumpToIndex(index) {
  const playlist = state.player.playlist;
  if (!playlist.length) return;
  const item = playlist[index];
  state.player.index = index;
  state.player.offset = Math.max(0, Number(item.start_time) - state.player.baseTime);
  clearActiveTurn();
  if (state.player.playing) {
    startTimeline(state.player.offset);
  } else {
    highlightActiveTurn(item.utterance_id);
    renderPlayer();
  }
}

function clearTimelinePlayback() {
  state.player.timers.forEach((timer) => window.clearTimeout(timer));
  state.player.timers = [];
  if (state.player.progressTimer) {
    window.clearInterval(state.player.progressTimer);
    state.player.progressTimer = null;
  }
  state.player.activeAudios.forEach((audio) => {
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
  });
  state.player.activeAudios = [];
}

function finishTimeline() {
  clearTimelinePlayback();
  state.player.offset = state.player.duration;
  state.player.index = Math.max(0, state.player.playlist.length - 1);
  state.player.playing = false;
  renderPlayer("Finished dialogue playback.");
  clearActiveTurn();
}

function currentOffset() {
  if (!state.player.playing) return state.player.offset;
  return clamp(
    state.player.startedOffset + ((performance.now() - state.player.startedAt) / 1000),
    0,
    Math.max(0, state.player.duration)
  );
}

function currentTimelineIndex(offset = currentOffset()) {
  const playlist = state.player.playlist;
  if (!playlist.length) return 0;
  let index = 0;
  for (let i = 0; i < playlist.length; i += 1) {
    const startOffset = Number(playlist[i].start_time) - state.player.baseTime;
    if (startOffset <= offset + 0.001) {
      index = i;
    } else {
      break;
    }
  }
  return index;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function renderPlayer(message = "") {
  const playlist = state.player.playlist;
  const hasDialogue = state.dialogue !== "all";
  const canPlay = hasDialogue && playlist.length > 0;
  const offset = currentOffset();
  const activeIndex = canPlay ? currentTimelineIndex(offset) : 0;
  const item = playlist[activeIndex];

  nodes.playDialogueButton.disabled = !canPlay;
  nodes.prevTurnButton.disabled = !canPlay || activeIndex <= 0;
  nodes.nextTurnButton.disabled = !canPlay || activeIndex >= playlist.length - 1;
  nodes.stopDialogueButton.disabled = !canPlay;
  nodes.playDialogueButton.textContent = state.player.playing ? "Pause" : "Play";

  if (!hasDialogue) {
    nodes.playerTitle.textContent = "Choose one dialogue";
    nodes.playerSubtitle.textContent = "Playback is available after selecting a dialogue.";
    nodes.playerProgress.style.width = "0%";
    return;
  }

  if (!playlist.length) {
    nodes.playerTitle.textContent = state.dialogue;
    nodes.playerSubtitle.textContent = message || "No timestamped audio files available for this dialogue.";
    nodes.playerProgress.style.width = "0%";
    return;
  }

  const progress = state.player.duration > 0 ? (offset / state.player.duration) * 100 : 0;
  nodes.playerTitle.textContent = state.dialogue;
  nodes.playerSubtitle.textContent = message || `${formatTime(offset)} / ${formatTime(state.player.duration)} · turn ${activeIndex + 1}/${playlist.length} · ${item.speaker_id} · ${item.transcript || item.utterance_id}`;
  nodes.playerProgress.style.width = `${clamp(progress, 0, 100)}%`;
}

function highlightActiveTurn(utteranceId) {
  clearActiveTurn();
  const node = document.getElementById(`turn-${utteranceId}`);
  if (!node) return;
  node.classList.add("playing");
  node.scrollIntoView({ behavior: "smooth", block: "center" });
}

function clearActiveTurn() {
  document.querySelectorAll(".utterance.playing").forEach((node) => {
    node.classList.remove("playing");
  });
}

function renderPrediction(item, prediction) {
  const isMatch = prediction.label === item.gold_label;
  const confidence = Math.round(prediction.confidence * 100);
  return `
    <div class="prediction">
      <div class="label-row">
        <span class="badge ${prediction.label}">${prediction.label}</span>
        <span>${confidence}%</span>
        <span class="${isMatch ? "match" : "mismatch"}">${isMatch ? "correct" : `gold: ${escapeHtml(item.gold_label)}`}</span>
      </div>
      <div class="prob-list">
        ${state.data.labels.map((label) => renderProbability(label, prediction.probabilities[label])).join("")}
      </div>
      ${renderModelComparison(item)}
    </div>
  `;
}

function renderModelComparison(item) {
  const models = ["baseline", "mal", "tim"];
  const rows = models
    .map((model) => {
      const pred = item.predictions[model];
      if (!pred) return "";
      const correct = pred.label === item.gold_label;
      return `<span class="model-pill ${correct ? "correct" : "wrong"}">${model}: ${pred.label}</span>`;
    })
    .join("");
  const proof = isTimFix(item) ? `<span class="proof-pill">${proofText(item.comparison.outcome)}</span>` : "";
  const paired = pairedOutcomeText(item.comparison?.mal_tim_outcome);
  const pairedPill = paired ? `<span class="pair-pill ${item.comparison.mal_tim_outcome}">${paired}</span>` : "";
  return `<div class="model-comparison">${rows}${proof}${pairedPill}</div>`;
}

function proofText(outcome) {
  const labels = {
    tim_correct_baseline_mal_wrong: "TIM fixes both",
    tim_correct_baseline_wrong: "TIM fixes baseline",
    tim_correct_mal_wrong: "TIM fixes MAL",
  };
  return labels[outcome] || "";
}

function pairedOutcomeText(outcome) {
  const labels = {
    tim_only_correct: "TIM correct, MAL wrong",
    mal_only_correct: "MAL correct, TIM wrong",
    both_correct: "MAL and TIM correct",
    both_wrong: "MAL and TIM wrong",
  };
  return labels[outcome] || "";
}

function renderMissingPrediction(item) {
  return `
    <div class="prediction muted-prediction">
      <div class="label-row">
        <span class="badge missing">no prediction</span>
        <span>gold: ${escapeHtml(item.gold_label || "unknown")}</span>
      </div>
      <p>Audio and timing are kept in the dialogue timeline; only model labels are missing for this utterance.</p>
    </div>
  `;
}

function renderProbability(label, value) {
  const percentage = Math.round((value || 0) * 100);
  return `
    <div class="prob-row">
      <span>${label}</span>
      <span class="prob-track"><span class="prob-fill" style="width: ${percentage}%; background: ${colorVars[label]}"></span></span>
      <span>${percentage}%</span>
    </div>
  `;
}

function getDialoguesForSession() {
  if (state.session === "all") {
    return Object.values(state.data.dialogues_by_session).flat();
  }
  return state.data.dialogues_by_session[state.session] || [];
}

function countForDialogues(dialogues) {
  return dialogues.reduce((total, dialogue) => total + dialogue.count, 0);
}

function formatTime(value) {
  if (!Number.isFinite(Number(value))) return "0.0s";
  return `${Number(value).toFixed(1)}s`;
}

function formatTurnMeta(item) {
  const turn = Number.isInteger(item.turn_index) ? `turn ${item.turn_index}` : "turn";
  if (item.start_time === null || item.end_time === null) {
    return `${turn} · ${Number(item.duration || 0).toFixed(1)}s`;
  }
  return `${turn} · ${formatTime(item.start_time)} - ${formatTime(item.end_time)}`;
}

function percent(value, total) {
  if (!total) return "0%";
  return `${Math.round((value / total) * 100)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}
