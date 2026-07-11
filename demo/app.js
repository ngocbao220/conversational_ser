const state = {
  data: null,
  session: "Ses05",
  scope: "session",
  backbone: "wavlm",
  dialogue: "all",
  model: "cim",
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

const modelColors = {
  baseline: "#64748b",
  cdm: "#d97706",
  cim: "#0f766e",
};

const nodes = {
  sessionFilters: document.querySelector("#sessionFilters"),
  backboneControls: document.querySelector("#backboneControls"),
  scopeControls: document.querySelector("#scopeControls"),
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
  { id: "full", name: "Full dialogue", help: "All turns, preserving context" },
  { id: "evaluated", name: "Evaluated only", help: "Only utterances with model predictions" },
  { id: "cim_fixes", name: "CDIM fixes only", help: "CDIM correct while baseline or CDM is wrong" },
];

const scopes = [
  { id: "session", name: "Session", help: "Show all turns in the selected session" },
  { id: "dialogue", name: "Dialogue", help: "Focus on one full conversation" },
];

const pairedOutcomes = [
  { id: "all", name: "All paired outcomes", help: "Keep every turn in the current view" },
  { id: "cim_correct_cdm_wrong", name: "CDIM correct, CDM wrong", help: "CDIM improves this turn over CDM" },
  { id: "cdm_correct_cim_wrong", name: "CDM correct, CDIM wrong", help: "CDM handles this turn better than CDIM" },
  { id: "both_correct", name: "Both correct", help: "CDM and CDIM match the gold label" },
  { id: "both_wrong", name: "Both wrong", help: "Both models miss the gold label" },
];

fetch(`demo_data.json?v=${Date.now()}`, { cache: "no-store" })
  .then((response) => response.json())
  .then((data) => {
    state.data = data;
    state.backbone = data.default_backbone || data.backbones?.[0]?.id || state.backbone;
    if (!modelsForBackbone().some((model) => model.id === state.model)) {
      state.model = modelsForBackbone()[0]?.id || "";
    }
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
  state.scope = state.dialogue === "all" ? "session" : "dialogue";
  stopDialogue();
  renderScopeControls();
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

nodes.streamList.addEventListener("click", (event) => {
  const interactive = event.target.closest("button, audio, input, select, a, label, summary");
  if (interactive) return;

  const utteranceNode = event.target.closest(".utterance[data-utterance-id]");
  if (!utteranceNode) return;
  selectUtteranceForPlayback(utteranceNode.dataset.utteranceId);
});

function renderControls() {
  renderSessionFilters();
  renderBackboneControls();
  renderScopeControls();
  renderDialogueOptions();
  renderViewControls();
  renderPairControls();
  renderModelControls();
  renderLabelFilters();
}

function renderScopeControls() {
  nodes.scopeControls.innerHTML = scopes
    .map((scope) => {
      const active = state.scope === scope.id ? " active" : "";
      return `<button class="segment${active}" data-scope="${scope.id}"><span>${scope.name}</span><small>${scope.help}</small></button>`;
    })
    .join("");

  nodes.scopeControls.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.scope = button.dataset.scope;
      if (state.scope === "session") {
        state.dialogue = "all";
      } else {
        state.dialogue = firstDialogueIdForSession() || "all";
      }
      stopDialogue();
      renderScopeControls();
      renderDialogueOptions();
      render();
    });
  });
}

function renderBackboneControls() {
  const backbones = state.data.backbones || [{ id: state.backbone, name: state.backbone }];
  nodes.backboneControls.innerHTML = backbones
    .map((backbone) => {
      const active = state.backbone === backbone.id ? " active" : "";
      return `<button class="segment${active}" data-backbone="${backbone.id}">${escapeHtml(backbone.name)}</button>`;
    })
    .join("");

  nodes.backboneControls.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.backbone = button.dataset.backbone;
      if (!modelsForBackbone().some((model) => model.id === state.model)) {
        state.model = modelsForBackbone()[0]?.id || "";
      }
      stopDialogue();
      renderBackboneControls();
      renderModelControls();
      renderPairControls();
      renderDialogueOptions();
      render();
    });
  });
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
      state.dialogue = state.scope === "dialogue" ? (firstDialogueIdForSession() || "all") : "all";
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
      const predicted = dialogue.predicted_count_by_backbone?.[state.backbone] ?? dialogue.predicted_count ?? 0;
      return `<option value="${dialogue.id}">${dialogue.id} (${dialogue.count} turns · ${predicted} predicted)</option>`;
    }),
  ].join("");
  if (state.scope === "dialogue" && state.dialogue === "all") {
    state.dialogue = firstDialogueIdForSession() || "all";
  }
  if (state.scope === "session") {
    state.dialogue = "all";
  }
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
  const counts = summaryForBackbone().cdm_cim_paired_counts || {};
  nodes.pairControls.innerHTML = pairedOutcomes
    .map((outcome) => {
      const active = state.pairedOutcome === outcome.id ? " active" : "";
      const count = outcome.id === "all"
        ? summaryForBackbone().fully_compared_count
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
  nodes.modelControls.innerHTML = modelsForBackbone()
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
  const activeModel = modelsForBackbone().find((model) => model.id === state.model);
  const predictedCount = utterances.filter((item) => !item.is_ignored_label && activePredictions(item)[state.model]).length;
  nodes.activeDialogue.textContent = state.scope === "dialogue" && state.dialogue !== "all"
    ? state.dialogue
    : (state.session === "all" ? "All sessions" : `${state.session} dialogues`);
  nodes.streamMeta.textContent = `${utterances.length.toLocaleString()} turns shown · ${predictedCount.toLocaleString()} have ${(activeModel?.name || "selected model")} predictions`;
  renderLabelBars(utterances);
  renderEvidencePanel(utterances);
  renderStream(utterances);
  syncPlayerPlaylist();
  renderPlayer();
}

function filteredUtterances() {
  return state.data.utterances.filter((item) => {
    const prediction = activePredictions(item)[state.model];
    if (state.session !== "all" && item.session_id !== state.session) return false;
    if (state.scope === "dialogue" && state.dialogue !== "all" && item.dialogue_id !== state.dialogue) return false;
    if (state.view === "evaluated" && (item.is_ignored_label || !prediction)) return false;
    if (state.view === "cim_fixes" && !isCimFix(item)) return false;
    if (state.dialogue === "all" && state.pairedOutcome !== "all" && activeComparison(item)?.cdm_cim_outcome !== state.pairedOutcome) return false;
    if (state.labels.size > 0 && (item.is_ignored_label || !prediction || !state.labels.has(prediction.label))) return false;
    if (!state.search) return true;
    const haystack = [
      item.utterance_id,
      item.dialogue_id,
      item.speaker_id,
      item.transcript,
      item.gold_label,
      item.mapped_label,
      item.raw_label,
      item.raw_emotion,
      item.raw_emotion_full,
      item.is_ignored_label ? "ignore ignored excluded" : "",
      prediction?.label || "",
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(state.search);
  });
}

function isCimFix(item) {
  const comparison = activeComparison(item) || {};
  return comparison.outcome === "cim_correct_baseline_cdm_wrong"
    || comparison.outcome === "cim_correct_baseline_wrong"
    || comparison.outcome === "cim_correct_cdm_wrong";
}

function renderLabelBars(utterances) {
  const counts = Object.fromEntries(state.data.labels.map((label) => [label, 0]));
  utterances.forEach((item) => {
    if (item.is_ignored_label) return;
    const label = activePredictions(item)[state.model]?.label;
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
  const compared = utterances.filter((item) => activeComparison(item)?.has_all_predictions);
  const cimCorrect = compared.filter((item) => activeComparison(item).cim_correct).length;
  const baselineCorrect = compared.filter((item) => activeComparison(item).baseline_correct).length;
  const cdmCorrect = compared.filter((item) => activeComparison(item).cdm_correct).length;
  const cimFixBaseline = compared.filter((item) => activeComparison(item).outcome === "cim_correct_baseline_wrong" || activeComparison(item).outcome === "cim_correct_baseline_cdm_wrong").length;
  const cimFixCdm = compared.filter((item) => activeComparison(item).outcome === "cim_correct_cdm_wrong" || activeComparison(item).outcome === "cim_correct_baseline_cdm_wrong").length;
  const cimFixBoth = compared.filter((item) => activeComparison(item).outcome === "cim_correct_baseline_cdm_wrong").length;

  nodes.evidencePanel.innerHTML = [
    renderEvidenceMetric("Compared", compared.length, "utterances with all 3 predictions"),
    renderEvidenceMetric("CDIM correct", cimCorrect, percent(cimCorrect, compared.length)),
    renderEvidenceMetric("Baseline correct", baselineCorrect, percent(baselineCorrect, compared.length)),
    renderEvidenceMetric("CDM correct", cdmCorrect, percent(cdmCorrect, compared.length)),
    renderEvidenceMetric("CDIM fixes baseline", cimFixBaseline, "baseline wrong, CDIM correct"),
    renderEvidenceMetric("CDIM fixes CDM", cimFixCdm, "CDM wrong, CDIM correct"),
    renderEvidenceMetric("CDIM fixes both", cimFixBoth, "baseline and CDM wrong"),
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
  const prediction = activePredictions(item)[state.model];
  const audio = item.audio_path
    ? `<audio class="audio" controls preload="none" src="../${escapeAttribute(item.audio_path)}"></audio>`
    : "";
  const labelClass = item.is_ignored_label ? "ignored" : (prediction?.label || "unpredicted");
  const proofClass = isCimFix(item) ? " cim-proof" : "";

  return `
    <article class="utterance ${labelClass}${proofClass}" id="turn-${escapeAttribute(item.utterance_id)}" data-utterance-id="${escapeAttribute(item.utterance_id)}">
      ${renderInteractionFeatureToggle(item)}
      <div class="utterance-main">
        <div class="speaker">
          <strong>${escapeHtml(item.speaker_id || "speaker")}</strong>
          <span>${escapeHtml(item.utterance_id)}</span>
          <span>${formatTurnMeta(item)}</span>
        </div>
        <div class="transcript">
          <p>${escapeHtml(item.transcript || "(no transcript)")}</p>
          ${audio}
        </div>
      </div>
      ${renderPredictionPanel(item, prediction)}
      ${renderSoftmaxHistogram(item)}
    </article>
  `;
}

function renderInteractionFeatureToggle(item) {
  const features = item.interaction_features || {};
  const rows = [
    ["Response timing", "relative_gap", features.relative_gap, "s vs this speaker's previous mean gap"],
    ["Floor competition", "overlap_ratio", features.overlap_ratio, "overlap / duration"],
    ["Duration", "duration", features.duration, "s"],
    ["Turn-taking", "speaker_switch", features.speaker_switch, ""],
  ];
  return `
    <details class="interaction-details">
      <summary>Interaction features</summary>
      <div class="interaction-grid">
        ${rows.map(([label, key, value, note]) => renderInteractionFeature(label, key, value, note)).join("")}
      </div>
    </details>
  `;
}

function renderInteractionFeature(label, key, value, note) {
  const shownValue = typeof value === "boolean" ? (value ? "yes" : "no") : Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "n/a";
  return `
    <div class="interaction-feature ${value === true ? "active" : ""}">
      <small>${escapeHtml(label)}</small>
      <strong>${escapeHtml(shownValue)}</strong>
      <span>${escapeHtml(key)}${note ? ` · ${escapeHtml(note)}` : ""}</span>
    </div>
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

function selectUtteranceForPlayback(utteranceId) {
  const item = state.data.utterances.find((row) => row.utterance_id === utteranceId);
  if (!item || item.start_time === null || item.end_time === null || !item.audio_path) {
    renderPlayer("This utterance has no timestamped audio.");
    return;
  }

  const wasPlaying = state.player.playing;
  if (state.dialogue !== item.dialogue_id) {
    clearTimelinePlayback();
    state.player.playing = false;
    state.dialogue = item.dialogue_id;
    renderDialogueOptions();
    render();
  } else if (wasPlaying) {
    clearTimelinePlayback();
    state.player.playing = false;
  }

  syncPlayerPlaylist();
  const index = state.player.playlist.findIndex((row) => row.utterance_id === utteranceId);
  if (index < 0) {
    renderPlayer("This utterance is outside the active playback playlist.");
    return;
  }

  const selected = state.player.playlist[index];
  state.player.index = index;
  state.player.offset = Math.max(0, Number(selected.start_time) - state.player.baseTime);
  highlightActiveTurn(selected.utterance_id);
  renderPlayer(`Ready from ${selected.utterance_id}. Press Play to continue from this turn.`);

  if (wasPlaying) {
    startTimeline(state.player.offset);
  }
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

function renderPredictionPanel(item, activePrediction) {
  return `
    <div class="prediction">
      ${renderEmotionSummary(item)}
      ${item.is_ignored_label || !activePrediction ? renderMissingPrediction(item) : ""}
    </div>
  `;
}

function renderEmotionSummary(item) {
  const raw = formatRawEmotion(item);
  const gold = item.is_ignored_label ? "ignore" : (item.gold_label || "unknown");
  const badgeClass = item.is_ignored_label ? "ignore" : (item.gold_label || "missing");
  const note = item.is_ignored_label
    ? "excluded from training"
    : (item.raw_emotion_full && item.gold_label && item.raw_emotion_full !== item.gold_label ? `mapped to ${item.gold_label}` : "target label");
  return `
    <div class="emotion-summary">
      <span class="emotion-token ${item.is_ignored_label ? "ignored" : ""}">
        <small>raw</small>
        <strong>${escapeHtml(raw)}</strong>
        <em>${escapeHtml(note)}</em>
      </span>
      <span class="emotion-token ${item.is_ignored_label ? "ignored" : ""}">
        <small>ground truth</small>
        <strong><span class="badge small ${badgeClass}">${escapeHtml(gold)}</span></strong>
      </span>
    </div>
  `;
}

function formatRawEmotion(item) {
  const rawCode = item.raw_label || item.raw_emotion || "";
  const rawFull = item.raw_emotion_full || "";
  if (rawCode && rawFull && rawCode !== rawFull) return `${rawCode} -> ${rawFull}`;
  return rawFull || rawCode || "unknown";
}

function renderSoftmaxHistogram(item) {
  if (item.is_ignored_label) return "";
  const predictions = activePredictions(item);
  const hasPrediction = modelsForBackbone().some((model) => predictions[model.id]);
  if (!hasPrediction) return "";
  return `
    <div class="softmax-histogram" aria-label="Softmax probabilities by model">
      <div class="histogram-head">
        <span>Model softmax histograms</span>
        <span>ground truth: ${escapeHtml(item.is_ignored_label ? "ignore" : (item.gold_label || "unknown"))}</span>
      </div>
      <div class="model-histogram-grid">
        ${modelsForBackbone().map((model) => renderModelHistogram(item, model)).join("")}
      </div>
    </div>
  `;
}

function renderModelHistogram(item, model) {
  const prediction = activePredictions(item)[model.id];
  if (!prediction) {
    return `
      <div class="model-histogram-card missing">
        <header>
          <span><span class="model-dot" style="background:${modelColors[model.id] || "#475569"}"></span>${escapeHtml(model.name)}</span>
          <strong>missing</strong>
        </header>
      </div>
    `;
  }
  const isIgnored = Boolean(item.is_ignored_label);
  const correct = !isIgnored && prediction.label === item.gold_label;
  const cardClass = isIgnored ? "ignored" : (correct ? "correct" : "wrong");
  const labelClass = isIgnored ? "ignored" : (correct ? "correct" : "wrong");
  return `
    <div class="model-histogram-card ${cardClass}">
      <header>
        <span><span class="model-dot" style="background:${modelColors[model.id] || "#475569"}"></span>${escapeHtml(model.name)}</span>
        <strong class="${labelClass}">${escapeHtml(prediction.label || "unknown")}</strong>
      </header>
      <div class="model-histogram-bars">
        ${state.data.labels.map((label) => renderModelEmotionBar(prediction, model, label)).join("")}
      </div>
    </div>
  `;
}

function renderModelEmotionBar(prediction, model, label) {
  const value = prediction?.probabilities?.[label] || 0;
  const percentage = Math.round(value * 100);
  const activeClass = prediction?.label === label ? " predicted" : "";
  return `
    <div class="model-prob${activeClass}" title="${escapeAttribute(model.name)} ${label}: ${percentage}%">
      <span class="histogram-label ${label}">${label}</span>
      <span class="model-prob-fill" style="width:calc((100% - 76px) * ${percentage} / 100); background:${modelColors[model.id] || "#475569"}"></span>
      <span class="model-prob-text">${percentage}%</span>
    </div>
  `;
}

function renderMissingPrediction(item) {
  if (item.is_ignored_label) {
    return `
      <div class="muted-prediction ignored">
        <div class="label-row">
          <span class="badge ignore">ignore</span>
          <span>raw: ${escapeHtml(formatRawEmotion(item))}</span>
        </div>
        <p>This raw emotion is excluded by the current training label mapping, but the utterance is kept for dialogue context.</p>
      </div>
    `;
  }
  return `
    <div class="muted-prediction">
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

function firstDialogueIdForSession() {
  return getDialoguesForSession()[0]?.id || "";
}

function modelsForBackbone() {
  return state.data?.models_by_backbone?.[state.backbone] || state.data?.models || [];
}

function summaryForBackbone() {
  return state.data?.summary?.by_backbone?.[state.backbone] || state.data?.summary || {};
}

function activePredictions(item) {
  return item.predictions_by_backbone?.[state.backbone] || item.predictions || {};
}

function activeComparison(item) {
  return item.comparison_by_backbone?.[state.backbone] || item.comparison || {};
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
