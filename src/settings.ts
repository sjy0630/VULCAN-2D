// ============================================================================
//  Settings UI — language selection and preferences
// ============================================================================

import { getCurrentLanguage, setLanguage, t, type Language } from './i18n';

export class Settings {
  private settingsBtn: HTMLElement;
  private settingsPanel: HTMLElement | null = null;
  private isOpen = false;

  constructor() {
    this.settingsBtn = this.createSettingsButton();
    this.attachEventListeners();
    // keep the panel's own labels in sync when the language changes
    window.addEventListener('language-changed', () => this.retranslate());
  }

  private retranslate(): void {
    if (!this.settingsPanel) return;
    const h3 = this.settingsPanel.querySelector('.settings-header h3');
    if (h3) h3.textContent = t('settings');
    const label = this.settingsPanel.querySelector('.settings-label');
    if (label) label.textContent = t('language');
    this.updatePanelUI();
  }

  private createSettingsButton(): HTMLElement {
    const container = document.createElement('div');
    container.className = 'settings-container';

    const btn = document.createElement('button');
    btn.className = 'settings-btn';
    btn.setAttribute('aria-label', 'Settings');
    btn.innerHTML = '⚙️';

    container.appendChild(btn);
    return container;
  }

  private createSettingsPanel(): HTMLElement {
    const panel = document.createElement('div');
    panel.className = 'settings-panel';

    const header = document.createElement('div');
    header.className = 'settings-header';
    header.innerHTML = `<h3>${t('settings')}</h3>`;

    const content = document.createElement('div');
    content.className = 'settings-content';

    const langSection = document.createElement('div');
    langSection.className = 'settings-section';

    const langLabel = document.createElement('label');
    langLabel.className = 'settings-label';
    langLabel.textContent = t('language');

    const langOptions = document.createElement('div');
    langOptions.className = 'settings-options';

    const currentLang = getCurrentLanguage();

    const enBtn = document.createElement('button');
    enBtn.className = `settings-option ${currentLang === 'en' ? 'active' : ''}`;
    enBtn.textContent = 'English';
    enBtn.addEventListener('click', () => this.selectLanguage('en'));

    const zhBtn = document.createElement('button');
    zhBtn.className = `settings-option ${currentLang === 'zh' ? 'active' : ''}`;
    zhBtn.textContent = '简体中文';
    zhBtn.addEventListener('click', () => this.selectLanguage('zh'));

    langOptions.appendChild(enBtn);
    langOptions.appendChild(zhBtn);

    langSection.appendChild(langLabel);
    langSection.appendChild(langOptions);

    content.appendChild(langSection);
    panel.appendChild(header);
    panel.appendChild(content);

    return panel;
  }

  private selectLanguage(lang: Language): void {
    setLanguage(lang);
    this.updatePanelUI();
  }

  private updatePanelUI(): void {
    if (!this.settingsPanel) return;

    const buttons = this.settingsPanel.querySelectorAll('.settings-option');
    const currentLang = getCurrentLanguage();

    buttons.forEach((btn, idx) => {
      const isActive = (idx === 0 && currentLang === 'en') || (idx === 1 && currentLang === 'zh');
      if (isActive) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });
  }

  private attachEventListeners(): void {
    const btn = this.settingsBtn.querySelector('button');
    if (btn) {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.togglePanel();
      });
    }

    document.addEventListener('click', () => {
      if (this.isOpen) {
        this.closePanel();
      }
    });

    this.settingsBtn.addEventListener('click', (e) => {
      e.stopPropagation();
    });
  }

  private togglePanel(): void {
    if (this.isOpen) {
      this.closePanel();
    } else {
      this.openPanel();
    }
  }

  private openPanel(): void {
    if (!this.settingsPanel) {
      this.settingsPanel = this.createSettingsPanel();
      this.settingsBtn.appendChild(this.settingsPanel);
    }
    this.settingsPanel.classList.add('open');
    this.isOpen = true;
  }

  private closePanel(): void {
    if (this.settingsPanel) {
      this.settingsPanel.classList.remove('open');
    }
    this.isOpen = false;
  }

  public getElement(): HTMLElement {
    return this.settingsBtn;
  }
}
