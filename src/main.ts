// ============================================================================
//  VULCAN-2D — application entry / orchestration.
//
//  The validated physics runs in the Python engine (vulcan2d.serve); this UI
//  fetches simulated + measured I-V loops and drives:
//    · left  — the non-filamentary 3D device (patches light up with phi_bar),
//    · right — the model-vs-measured I-V overlay + a validation readout.
// ============================================================================

import './style.css';
import { simulate, ping, type SimResult, type SimParams } from './engine';
import { IVPlot, STYLE_MODEL, STYLE_DATA } from './viz/ivplot';
import { Device3D } from './viz/device3d';
import { applyTranslations, t } from './i18n';
import { Settings } from './settings';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const deviceCanvas = $<HTMLCanvasElement>('device-canvas');
const ivCanvas = $<HTMLCanvasElement>('iv-canvas');

const layers = $<HTMLInputElement>('layers');
const layersVal = $('layers-val');
const thickVal = $('thick-val');
const kEl = $<HTMLInputElement>('k');
const kVal = $('k-val');
const icc = $<HTMLInputElement>('icc');
const iccVal = $('icc-val');
const sig = $<HTMLInputElement>('sig');
const sigVal = $('sig-val');
const cyc = $<HTMLInputElement>('cyc');
const cycVal = $('cyc-val');
const runBtn = $<HTMLButtonElement>('run');

const chipEngine = $('chip-engine');
const chipCyc = $('chip-cyc');
const chipFps = $('chip-fps');
const ivHint = $('iv-hint');

const device = new Device3D(deviceCanvas);
const plot = new IVPlot(ivCanvas, -2, 5);

let seed = 2026;
let needsSim = true;
let pending = false;
let engineUp = false;
let lastNcycles: number | null = null;

// Re-render the dynamically-managed chips (no data-i18n on them) so they show
// live values in the current language; called on boot and on language change.
function refreshDynamicChips() {
  chipEngine.textContent = engineUp ? t('engine-ok') : t('engine-offline');
  chipEngine.className = engineUp ? 'chip up' : 'chip down';
  chipCyc.textContent = `${lastNcycles ?? '—'} ${t('chip-cyc')}`;
}

function params(): SimParams {
  return {
    icc_ua: parseFloat(icc.value),
    K: parseInt(kEl.value, 10),
    sigma: parseFloat(sig.value),
    ncycles: parseInt(cyc.value, 10),
    seed,
  };
}

async function runSim() {
  if (pending) return;
  pending = true;
  ivHint.textContent = t('computing');
  try {
    const r = await simulate(params());
    applyResult(r);
    engineUp = true;
  } catch (e) {
    engineUp = false;
    ivHint.textContent = t('start-hint');
  } finally {
    refreshDynamicChips();
    pending = false;
  }
}

function applyResult(r: SimResult) {
  plot.setSeries([
    { set: r.measuredSet, reset: r.measuredReset, style: STYLE_DATA },
    { set: r.setLoop, reset: r.resetLoop, style: STYLE_MODEL },
  ]);
  device.setPatchCount(r.K);
  device.triggerBreakdown(r.features.phi_final);
  fillFeatures(r);
  lastNcycles = r.ncycles;
  chipCyc.textContent = `${r.ncycles} ${t('chip-cyc')}`;
  ivHint.textContent = '';
}

// ---- validation readout (model vs measured target) ----
function fillFeatures(r: SimResult) {
  const f = r.features;
  const t = r.targets;
  setFeat('f-vset', `${f.Vset.mean.toFixed(2)} V`, `${t.Vset.toFixed(2)}`, rel(f.Vset.mean, t.Vset));
  setFeat('f-vreset', `${f.Vreset.mean.toFixed(2)} V`, `${t.Vreset.toFixed(2)}`, rel(f.Vreset.mean, t.Vreset));
  setFeat('f-rhrs', sci(f.R_HRS.mean), sci(t.R_HRS), rel(f.R_HRS.mean, t.R_HRS));
  setFeat('f-rlrs', sci(f.R_LRS.mean), sci(t.R_LRS), rel(f.R_LRS.mean, t.R_LRS));
  setFeat('f-icc', `${(f.Icc.mean * 1e6).toFixed(1)} µA`, `${(t.Icc * 1e6).toFixed(0)}`, rel(f.Icc.mean, t.Icc));
  setFeat('f-win', `${f.window.toFixed(0)}×`, `${t.window.toFixed(0)}`, rel(f.window, t.window));
}
function setFeat(id: string, model: string, data: string, ok: boolean) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = `<span class="${ok ? 'ok' : 'off'}">${model}</span> <span class="data">/ ${data}</span>`;
}
function rel(a: number, b: number): boolean {
  return Math.abs(Math.abs(a) - Math.abs(b)) <= 0.35 * Math.abs(b);
}
function sci(x: number): string {
  const e = Math.floor(Math.log10(Math.abs(x)));
  const m = x / Math.pow(10, e);
  return `${m.toFixed(1)}e${e}`;
}

// ---- labels ----
function syncLabels() {
  layersVal.textContent = layers.value;
  thickVal.textContent = `${(parseInt(layers.value, 10) * 0.333).toFixed(2)} nm`;
  kVal.textContent = kEl.value;
  iccVal.textContent = `${icc.value} µA`;
  sigVal.textContent = parseFloat(sig.value).toFixed(2);
  cycVal.textContent = cyc.value;
}
function applyGeometry() {
  device.setGeometry({ hbnLayers: parseInt(layers.value, 10) });
}

const markDirty = () => {
  needsSim = true;
};

layers.addEventListener('input', () => {
  syncLabels();
  applyGeometry();
}); // visual only
kEl.addEventListener('input', () => {
  syncLabels();
  markDirty();
});
icc.addEventListener('input', () => {
  syncLabels();
  markDirty();
});
sig.addEventListener('input', () => {
  syncLabels();
  markDirty();
});
cyc.addEventListener('input', () => {
  syncLabels();
  markDirty();
});
runBtn.addEventListener('click', () => {
  seed = (seed + 1) | 0;
  markDirty();
});

function onResize() {
  device.resize(deviceCanvas);
  plot.resize();
}
window.addEventListener('resize', onResize);

// ---- loop ----
let last = performance.now();
let fpsAccum = 0;
let fpsFrames = 0;
function animate(now: number) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (needsSim && !pending) {
    needsSim = false;
    void runSim();
  }
  device.render(dt);
  fpsAccum += dt;
  fpsFrames++;
  if (fpsAccum >= 0.5) {
    chipFps.textContent = `${Math.round(fpsFrames / fpsAccum)} fps`;
    fpsAccum = 0;
    fpsFrames = 0;
  }
  requestAnimationFrame(animate);
}

// ---- boot ----
syncLabels();
applyGeometry();
onResize();

const settings = new Settings();
const settingsContainer = $('settings-container');
settingsContainer.appendChild(settings.getElement());

applyTranslations();
refreshDynamicChips();

// keep the JS-managed chips / hint in sync when the language changes
window.addEventListener('language-changed', () => {
  refreshDynamicChips();
  if (!engineUp) ivHint.textContent = t('start-hint');
});

ping().then((up) => {
  engineUp = up;
  refreshDynamicChips();
});
requestAnimationFrame(animate);
