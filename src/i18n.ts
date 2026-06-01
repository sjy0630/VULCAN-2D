// ============================================================================
//  Internationalization (i18n) — language translations and locale management
// ============================================================================

export type Language = 'en' | 'zh';

const translations: Record<Language, Record<string, string>> = {
  en: {
    'brand-mark': 'VULCAN‐2D',
    'brand-sub': 'Layered-material memristor simulator · non-filamentary v0.2',
    'device-title': 'Device · soft-breakdown patches',
    'device-hint': 'drag to orbit',
    'iv-title': 'I–V · model vs measured',
    'layers-label': 'h-BN layers',
    'thick-foot': 'nm',
    'k-label': 'Sub-populations K',
    'k-foot': 'areal patches',
    'icc-label': 'Compliance I<sub>cc</sub>',
    'sig-label': 'Variability σ',
    'cyc-label': 'MC cycles',
    'run-btn': 'Re-sample',
    'feat-vset': 'V_set',
    'feat-vreset': 'V_reset',
    'feat-rhrs': 'R_HRS',
    'feat-rlrs': 'R_LRS',
    'feat-icc': 'I_cc',
    'feat-window': 'window',
    'chip-engine': 'engine',
    'chip-cyc': 'cycles',
    'chip-fps': 'fps',
    'computing': 'computing…',
    'engine-offline': 'engine offline',
    'start-hint': 'start: python -m vulcan2d.serve',
    'engine-ok': 'engine ✓',
    'settings': 'Settings',
    'language': 'Language',
  },
  zh: {
    'brand-mark': 'VULCAN‐2D',
    'brand-sub': '层状材料忆阻器仿真器 · 非细丝 v0.2',
    'device-title': '器件 · 软击穿斑块',
    'device-hint': '拖动以旋转',
    'iv-title': '伏安特性 · 模型 vs 实测',
    'layers-label': 'h-BN 层数',
    'thick-foot': 'nm',
    'k-label': '亚群数 K',
    'k-foot': '面内斑块',
    'icc-label': '限流 I<sub>cc</sub>',
    'sig-label': '变异性 σ',
    'cyc-label': '蒙特卡洛循环',
    'run-btn': '重新采样',
    'feat-vset': 'V_set',
    'feat-vreset': 'V_reset',
    'feat-rhrs': 'R_HRS',
    'feat-rlrs': 'R_LRS',
    'feat-icc': 'I_cc',
    'feat-window': 'window',
    'chip-engine': '引擎',
    'chip-cyc': '循环',
    'chip-fps': 'fps',
    'computing': '计算中…',
    'engine-offline': '引擎离线',
    'start-hint': '启动: python -m vulcan2d.serve',
    'engine-ok': '引擎 ✓',
    'settings': '设置',
    'language': '语言',
  },
};

let currentLang: Language = (localStorage.getItem('vulcan2d-lang') as Language) || 'en';

export function getCurrentLanguage(): Language {
  return currentLang;
}

export function setLanguage(lang: Language): void {
  if (currentLang !== lang) {
    currentLang = lang;
    localStorage.setItem('vulcan2d-lang', lang);
    window.dispatchEvent(new CustomEvent('language-changed', { detail: { lang } }));
  }
}

export function t(key: string): string {
  return translations[currentLang]?.[key] ?? key;
}

export function applyTranslations(): void {
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const key = el.getAttribute('data-i18n');
    if (key) {
      const val = t(key);
      if (el.tagName === 'INPUT' && el.getAttribute('type') === 'button') {
        el.setAttribute('value', val);
      } else if (el.hasAttribute('placeholder')) {
        el.setAttribute('placeholder', val);
      } else if (val.includes('<')) {
        // translation carries markup (e.g. a subscript) — render as HTML.
        // Safe: translation strings are developer-controlled, not user input.
        el.innerHTML = val;
      } else {
        el.textContent = val;
      }
    }
  });
}

window.addEventListener('language-changed', () => {
  applyTranslations();
});
