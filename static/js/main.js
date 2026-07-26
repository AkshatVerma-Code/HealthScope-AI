/* ============================================================
   main.js — LabSense AI Shared Interactions
   ============================================================ */

document.addEventListener('DOMContentLoaded', function () {

  // ── Navbar scroll effect ──────────────────────────────────
  const navbar = document.querySelector('.navbar');
  if (navbar) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 20) {
        navbar.style.background = 'rgba(10, 15, 30, 0.97)';
        navbar.style.borderBottomColor = 'rgba(0, 212, 170, 0.15)';
      } else {
        navbar.style.background = 'rgba(10, 15, 30, 0.85)';
        navbar.style.borderBottomColor = 'rgba(255, 255, 255, 0.08)';
      }
    });
  }

  // ── Scroll reveal animations ──────────────────────────────
  const revealElements = document.querySelectorAll(
    '.feature-card, .card, .choice-card, .risk-card, .animate-on-scroll'
  );

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, idx) => {
      if (entry.isIntersecting) {
        setTimeout(() => {
          entry.target.classList.add('animate-fade-in-up');
          entry.target.style.opacity = '1';
        }, idx * 80);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  revealElements.forEach(el => {
    el.style.opacity = '0';
    observer.observe(el);
  });

  // ── File Upload Zone ──────────────────────────────────────
  setupUploadZone('#upload-zone', '#report-file-input', '#upload-preview', '#preview-filename');
  setupUploadZone('#image-upload-zone', '#image-file-input', '#image-upload-preview', '#image-preview-filename');

  function setupUploadZone(zoneSelector, inputSelector, previewSelector, filenameSelector) {
    const zone     = document.querySelector(zoneSelector);
    const input    = document.querySelector(inputSelector);
    const preview  = document.querySelector(previewSelector);
    const filename = document.querySelector(filenameSelector);

    if (!zone || !input) return;

    // Drag events
    zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('drag-over');
      const files = e.dataTransfer.files;
      if (files.length) {
        const dt = new DataTransfer();
        dt.items.add(files[0]);
        input.files = dt.files;
        showPreview(files[0]);
      }
    });

    input.addEventListener('change', () => {
      if (input.files.length) showPreview(input.files[0]);
    });

    function showPreview(file) {
      if (filename) filename.textContent = file.name;
      if (preview)  preview.classList.add('visible');
    }
  }

  // ── Report type → dynamic fields ─────────────────────────
  const reportTypeSelect = document.getElementById('report-type-select');
  if (reportTypeSelect) {
    reportTypeSelect.addEventListener('change', () => {
      updateReportTypeHint(reportTypeSelect.value);
    });
    updateReportTypeHint(reportTypeSelect.value);
  }

  function updateReportTypeHint(type) {
    const hintEl = document.getElementById('report-type-hint');
    if (!hintEl) return;
    const hints = {
      'CBC':     '💉 CBC — Hemoglobin, WBC, RBC, Platelets, PCV',
      'LFT':     '🫀 LFT — ALT, AST, ALP, Bilirubin, Albumin',
      'KFT':     '🫘 KFT — Creatinine, Urea, Sodium, Potassium',
    };
    hintEl.textContent = hints[type] || '';
  }

  // ── Image type selector ───────────────────────────────────
  const imageTypeCards = document.querySelectorAll('.image-type-card');
  const imageTypeInput = document.getElementById('image-type-hidden');

  imageTypeCards.forEach(card => {
    card.addEventListener('click', () => {
      imageTypeCards.forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      if (imageTypeInput) imageTypeInput.value = card.dataset.type;
    });
  });

  // ── Form loading states ───────────────────────────────────
  const analysisForm = document.getElementById('analysis-form');
  if (analysisForm) {
    analysisForm.addEventListener('submit', (e) => {
      const submitBtn = analysisForm.querySelector('[type="submit"]');
      if (submitBtn) {
        submitBtn.innerHTML = '<span class="spinner-sm"></span> Analyzing...';
        submitBtn.disabled = true;
        submitBtn.style.opacity = '0.8';
      }
      // Show loading overlay
      const overlay = document.getElementById('loading-overlay');
      if (overlay) overlay.style.display = 'flex';
    });
  }

  // ── Smooth number counter animation ──────────────────────
  document.querySelectorAll('.counter-animate').forEach(el => {
    const target = parseInt(el.dataset.target, 10);
    const duration = 1800;
    const start = performance.now();

    function countUp(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(eased * target).toLocaleString();
      if (progress < 1) requestAnimationFrame(countUp);
    }

    // Start when visible
    const obs = new IntersectionObserver(entries => {
      if (entries[0].isIntersecting) {
        requestAnimationFrame(countUp);
        obs.disconnect();
      }
    });
    obs.observe(el);
  });

  // ── Print / Download shortcut ─────────────────────────────
  document.querySelectorAll('[data-action="print"]').forEach(btn => {
    btn.addEventListener('click', () => window.print());
  });

  // ── Tab Switcher ──────────────────────────────────────────
  document.querySelectorAll('[data-tab-target]').forEach(trigger => {
    trigger.addEventListener('click', () => {
      const target = trigger.dataset.tabTarget;
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('[data-tab-target]').forEach(t => t.classList.remove('active'));
      trigger.classList.add('active');
      const panel = document.getElementById(target);
      if (panel) panel.classList.add('active');
    });
  });

  // ── Auto-dismiss alerts ───────────────────────────────────
  document.querySelectorAll('.alert-auto-dismiss').forEach(alert => {
    setTimeout(() => {
      alert.style.opacity = '0';
      alert.style.transition = 'opacity 0.5s';
      setTimeout(() => alert.remove(), 500);
    }, 5000);
  });

});
