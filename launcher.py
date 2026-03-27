"""
G-Trade — GUI Launcher (Tkinter)
Графический лаунчер вместо bat-меню.
"""

import os
import sys
import codecs
import subprocess
import threading
import sqlite3
import tkinter as tk
from tkinter import ttk, scrolledtext

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
DB_PATH = os.path.join(BASE_DIR, "market.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Цвета ────────────────────────────────────────────────────────────────────
BG = "#0a0e17"
BG_CARD = "#111827"
BG_BUTTON = "#1e293b"
BG_BUTTON_HOVER = "#334155"
BG_BUTTON_ACTIVE = "#0ea5e9"
FG = "#e2e8f0"
FG_DIM = "#64748b"
FG_ACCENT = "#38bdf8"
FG_GREEN = "#22c55e"
FG_RED = "#ef4444"
FG_YELLOW = "#eab308"
FG_ORANGE = "#f97316"
BORDER = "#1e293b"


class LogRedirector:
    """Перенаправляет stdout/stderr в виджет Text."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, text):
        if not text:
            return
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, text, self.tag)
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")

    def flush(self):
        pass


class GTradeLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("G-TRADE — Control Center")
        self.configure(bg=BG)
        self.geometry("1060x820")
        self.minsize(900, 700)
        self._process = None
        self._running_task = None
        self._cr_line = False  # True если последняя строка лога — \r (progress bar)

        # Иконка окна (опционально)
        try:
            self.iconbitmap(default="")
        except Exception:
            pass

        self._build_ui()
        self._refresh_status()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # --- Верхняя панель (заголовок + статус) ---
        top = tk.Frame(self, bg=BG)
        top.pack(fill=tk.X, padx=16, pady=(12, 0))

        tk.Label(
            top, text="G-TRADE", font=("Consolas", 20, "bold"),
            fg=FG_ACCENT, bg=BG
        ).pack(side=tk.LEFT)

        tk.Label(
            top, text="Control Center", font=("Consolas", 12),
            fg=FG_DIM, bg=BG
        ).pack(side=tk.LEFT, padx=(10, 0), pady=(6, 0))

        self._status_label = tk.Label(
            top, text="", font=("Consolas", 10), fg=FG_DIM, bg=BG
        )
        self._status_label.pack(side=tk.RIGHT)

        # --- Разделитель ---
        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, padx=16, pady=8)

        # --- Основное тело: левая панель (кнопки) + правая (лог) ---
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Левая панель — кнопки со скроллом
        left_outer = tk.Frame(body, bg=BG, width=300)
        left_outer.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        left_outer.grid_propagate(False)

        left_canvas = tk.Canvas(left_outer, bg=BG, highlightthickness=0, width=280)
        left_scroll = tk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
        left = tk.Frame(left_canvas, bg=BG)

        left.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scroll.set)

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Прокрутка колесом мыши
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_buttons(left)

        # Правая панель — лог + статус
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Статус-карточки
        cards = tk.Frame(right, bg=BG)
        cards.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_status_cards(cards)

        # Лог-панель
        log_frame = tk.Frame(right, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self._log = scrolledtext.ScrolledText(
            log_frame, bg=BG_CARD, fg=FG, font=("Consolas", 9),
            insertbackground=FG, selectbackground="#334155",
            wrap=tk.WORD, state="disabled", borderwidth=0,
            highlightthickness=0
        )
        self._log.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._log.tag_configure("stdout", foreground=FG)
        self._log.tag_configure("stderr", foreground=FG_RED)
        self._log.tag_configure("info", foreground=FG_ACCENT)
        self._log.tag_configure("success", foreground=FG_GREEN)
        self._log.tag_configure("warn", foreground=FG_YELLOW)

        # Нижняя панель лога — кнопки
        log_bottom = tk.Frame(right, bg=BG)
        log_bottom.grid(row=2, column=0, sticky="ew", pady=(4, 0))

        self._btn_stop = self._make_small_btn(log_bottom, "STOP", self._stop_process, fg=FG_RED)
        self._btn_stop.pack(side=tk.RIGHT, padx=(4, 0))
        self._btn_stop.configure(state="disabled")

        self._make_small_btn(log_bottom, "Clear Log", self._clear_log).pack(side=tk.RIGHT, padx=(4, 0))
        self._make_small_btn(log_bottom, "Copy Log", self._copy_log).pack(side=tk.RIGHT, padx=(4, 0))

        self._task_label = tk.Label(
            log_bottom, text="Idle", font=("Consolas", 9),
            fg=FG_DIM, bg=BG
        )
        self._task_label.pack(side=tk.LEFT)

    def _build_buttons(self, parent):
        """Создаёт кнопки действий."""

        sections = [
            ("ОСНОВНОЕ", [
                (">  Full Cycle",       "Данные > Обучение > Dashboard", self._run_full),
                ("   Dashboard",        "Streamlit аналитика",          self._run_dashboard),
                ("   Predict (Radar)",  "Прогноз по текущим данным",    self._run_predict),
            ]),
            ("ДАННЫЕ & ОБУЧЕНИЕ", [
                ("   Data Update",      "Загрузить новые данные",       self._run_data),
                ("   Train Models",     "Обучение CatBoost + LSTM",    self._run_train),
                ("   Backtest",         "Тест стратегии на истории",    self._run_backtest),
            ]),
            ("АНАЛИТИКА", [
                ("   News Analyzer",    "Новости + sentiment",          self._run_news),
                ("   News Digest",      "Обзор авторитетных изданий",   self._run_digest),
                ("   Regime Detector",  "Рыночный режим",               self._run_regime),
                ("   Correlation",      "Корреляции + алерты",          self._run_corr),
                ("   Paper Trading",    "Виртуальный портфель",         self._run_paper),
                ("   Watchlist",        "Избранные активы",             self._run_watchlist),
                ("   Signal Radar",     "Все сигналы одним экраном",    self._run_signal_radar),
                ("   Sector Rotation",  "Секторная ротация",            self._run_sector_rotation),
                ("   What-If",          "Симуляция портфеля — 5 режимов", None, [
                    ("  ↳ Top-5  90д equal",  "Топ-5 по Score, 90 торг. дней",   self._run_whatif_top5),
                    ("  ↳ Top-10 90д equal",  "Топ-10 по Score, 90 торг. дней",  self._run_whatif_top10),
                    ("  ↳ Top-5 180д equal",  "Топ-5 по Score, полгода назад",   self._run_whatif_180),
                    ("  ↳ Top-5  90д Kelly",  "Веса пропорционально Score",      self._run_whatif_kelly),
                    ("  ↳ Custom assets",     "Свои активы + дни + капитал",     self._run_whatif_custom),
                ]),
                ("   Auto-Trader",      "Авто-исполнение сигналов",     self._run_auto_trader),
                ("   Check Alerts",     "Проверка пользовательских алертов", self._run_check_alerts),
                ("   Optuna Tune",      "Байесовская оптимизация гиперпараметров", self._run_optuna),
            ]),
            ("ОТЧЁТЫ & МОНИТОРИНГ", [
                ("   Model Health",     "Возраст/качество моделей",     self._run_model_health),
                ("   Model Compare",    "Сравнение моделей по обучениям", self._run_model_compare),
                ("   Performance",      "Точность ML прогнозов",        self._run_performance),
                ("   Guru Track",       "Точность совета легенд",       self._run_guru_track),
                ("   Guru Report",      "Фундаментал + анализ гуру",   self._run_guru_report),
                ("   Export Signals",   "Выгрузка сигналов CSV",        self._run_export),
                ("   Signal Log",       "История сигналов",             self._run_signal_log),
                ("   HTML Report",      "Полный отчёт в браузере",      self._run_report),
                ("   Equity Curve",     "График капитала",              self._run_equity),
            ]),
            ("СЕРВИСЫ", [
                ("   Telegram Bot",     "Sentinel V76",                 self._run_bot),
                ("   Scheduler",        "Автозапуск по расписанию",     self._run_scheduler),
                ("   DB Check",         "Проверка базы данных",         self._run_db_check),
                ("   DB Fix",           "Авто-ремонт базы данных",     self._run_db_fix),
                ("   DB Backup",        "Бэкап market.db",              self._run_backup),
                ("   Install/Repair",   "Установить зависимости",      self._run_install),
            ]),
        ]

        for section_idx, (title, buttons) in enumerate(sections):
            if section_idx > 0:
                tk.Frame(parent, bg=BG, height=8).pack()

            tk.Label(
                parent, text=title, font=("Consolas", 9, "bold"),
                fg=FG_DIM, bg=BG, anchor="w"
            ).pack(fill=tk.X, pady=(4, 4))

            for item in buttons:
                if len(item) == 4:
                    label, tooltip, _, sub_items = item
                    self._make_group_btn(parent, label, tooltip, sub_items)
                else:
                    label, tooltip, cmd = item
                    self._make_action_btn(parent, label, tooltip, cmd)

        # EXIT внизу
        tk.Frame(parent, bg=BG, height=16).pack()
        exit_btn = tk.Button(
            parent, text="EXIT", font=("Consolas", 11, "bold"),
            fg=FG_RED, bg=BG_BUTTON, activeforeground=FG, activebackground="#7f1d1d",
            relief="flat", cursor="hand2", padx=12, pady=6,
            command=self.destroy
        )
        exit_btn.pack(fill=tk.X, pady=2)
        exit_btn.bind("<Enter>", lambda e: exit_btn.configure(bg=BG_BUTTON_HOVER))
        exit_btn.bind("<Leave>", lambda e: exit_btn.configure(bg=BG_BUTTON))

    def _make_action_btn(self, parent, text, tooltip, command):
        """Создаёт стилизованную кнопку."""
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill=tk.X, pady=2)

        btn = tk.Button(
            frame, text=text, font=("Consolas", 11),
            fg=FG, bg=BG_BUTTON, activeforeground=FG, activebackground=BG_BUTTON_ACTIVE,
            relief="flat", cursor="hand2", anchor="w", padx=12, pady=6,
            command=command
        )
        btn.pack(fill=tk.X)

        tip = tk.Label(
            frame, text=f"  {tooltip}", font=("Consolas", 8),
            fg=FG_DIM, bg=BG, anchor="w"
        )
        tip.pack(fill=tk.X)

        btn.bind("<Enter>", lambda e: btn.configure(bg=BG_BUTTON_HOVER))
        btn.bind("<Leave>", lambda e: btn.configure(bg=BG_BUTTON))

        return btn

    def _make_group_btn(self, parent, text, tooltip, sub_items):
        """Collapsible button group: click to expand/collapse sub-buttons."""
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill=tk.X, pady=2)

        sub_frame = tk.Frame(outer, bg=BG)
        expanded = [False]

        def toggle():
            if expanded[0]:
                sub_frame.pack_forget()
                btn.configure(text=f"▶{text}")
            else:
                sub_frame.pack(fill=tk.X, padx=(16, 0))
                btn.configure(text=f"▼{text}")
            expanded[0] = not expanded[0]

        btn = tk.Button(
            outer, text=f"▶{text}", font=("Consolas", 11),
            fg=FG_YELLOW, bg=BG_BUTTON,
            activeforeground=FG, activebackground=BG_BUTTON_ACTIVE,
            relief="flat", cursor="hand2", anchor="w", padx=12, pady=6,
            command=toggle,
        )
        btn.pack(fill=tk.X)
        btn.bind("<Enter>", lambda e: btn.configure(bg=BG_BUTTON_HOVER))
        btn.bind("<Leave>", lambda e: btn.configure(bg=BG_BUTTON))

        tk.Label(
            outer, text=f"  {tooltip}", font=("Consolas", 8),
            fg=FG_DIM, bg=BG, anchor="w"
        ).pack(fill=tk.X)

        for sub_label, sub_tooltip, sub_cmd in sub_items:
            self._make_sub_btn(sub_frame, sub_label, sub_tooltip, sub_cmd)

        return btn

    def _make_sub_btn(self, parent, text, tooltip, command):
        """Indented sub-button inside a collapsible group."""
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill=tk.X, pady=1)

        btn = tk.Button(
            frame, text=text, font=("Consolas", 10),
            fg=FG_ACCENT, bg="#0f172a",
            activeforeground=FG, activebackground=BG_BUTTON_ACTIVE,
            relief="flat", cursor="hand2", anchor="w", padx=10, pady=4,
            command=command,
        )
        btn.pack(fill=tk.X)
        btn.bind("<Enter>", lambda e: btn.configure(bg=BG_BUTTON_HOVER))
        btn.bind("<Leave>", lambda e: btn.configure(bg="#0f172a"))

        tk.Label(
            frame, text=f"    {tooltip}", font=("Consolas", 8),
            fg=FG_DIM, bg=BG, anchor="w"
        ).pack(fill=tk.X)

        return btn

    def _make_small_btn(self, parent, text, command, fg=FG_DIM):
        btn = tk.Button(
            parent, text=text, font=("Consolas", 9),
            fg=fg, bg=BG_BUTTON, activeforeground=FG, activebackground=BG_BUTTON_HOVER,
            relief="flat", cursor="hand2", padx=8, pady=2,
            command=command
        )
        btn.bind("<Enter>", lambda e: btn.configure(bg=BG_BUTTON_HOVER))
        btn.bind("<Leave>", lambda e: btn.configure(bg=BG_BUTTON))
        return btn

    def _build_status_cards(self, parent):
        """Информационные карточки вверху."""
        cards_data = [
            ("db_size", "DB"),
            ("db_tables", "Tables"),
            ("models", "Models"),
            ("last_update", "Updated"),
            ("gpu", "GPU"),
        ]
        self._cards = {}
        for i, (key, title) in enumerate(cards_data):
            card = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0 if i == 0 else 4, 0))

            tk.Label(
                card, text=title, font=("Consolas", 8),
                fg=FG_DIM, bg=BG_CARD
            ).pack(anchor="w", padx=8, pady=(4, 0))

            val = tk.Label(
                card, text="—", font=("Consolas", 11, "bold"),
                fg=FG, bg=BG_CARD
            )
            val.pack(anchor="w", padx=8, pady=(0, 4))
            self._cards[key] = val

    # ── Статус ────────────────────────────────────────────────────────────

    def _refresh_status(self):
        """Обновляет статус-карточки."""
        try:
            # DB size
            if os.path.exists(DB_PATH):
                size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
                self._cards["db_size"].configure(text=f"{size_mb:.1f} MB")
            else:
                self._cards["db_size"].configure(text="N/A", fg=FG_RED)

            # Tables & last update
            if os.path.exists(DB_PATH):
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                tables = [r[0] for r in cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]
                self._cards["db_tables"].configure(text=str(len(tables)))

                # Последняя дата из первой таблицы
                max_date = None
                for t in tables[:5]:
                    try:
                        row = cur.execute(f"SELECT MAX(Date) FROM {t}").fetchone()
                        if row and row[0]:
                            d = row[0][:10]
                            if max_date is None or d > max_date:
                                max_date = d
                    except Exception:
                        pass
                conn.close()
                if max_date:
                    self._cards["last_update"].configure(text=max_date)

            # Models count
            if os.path.isdir(MODEL_DIR):
                cbm = len([f for f in os.listdir(MODEL_DIR) if f.endswith(".cbm")])
                keras = len([f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")])
                self._cards["models"].configure(text=f"{cbm}cb/{keras}ls")
            else:
                self._cards["models"].configure(text="0")

            # GPU
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if r.returncode == 0 and r.stdout.strip():
                    name = r.stdout.strip().split("\n")[0]
                    # Сокращаем имя
                    short = name.replace("NVIDIA ", "").replace("GeForce ", "")
                    self._cards["gpu"].configure(text=short, fg=FG_GREEN)
                else:
                    self._cards["gpu"].configure(text="CPU only", fg=FG_YELLOW)
            except Exception:
                self._cards["gpu"].configure(text="CPU only", fg=FG_YELLOW)

        except Exception as e:
            self._log_msg(f"Status error: {e}\n", "warn")

    # ── Логирование ───────────────────────────────────────────────────────

    def _log_msg(self, text, tag="info"):
        if not text:
            return
        self._log.configure(state="normal")
        if self._cr_line:
            self._log.insert(tk.END, "\n")
            self._cr_line = False
        self._log.insert(tk.END, text, tag)
        self._log.see(tk.END)
        self._log.configure(state="disabled")

    def _log_line(self, text, tag="stdout", is_cr=False):
        """Вставить строку в лог. is_cr=True → заменить предыдущую \r строку."""
        if not text:
            return
        self._log.configure(state="normal")
        if self._cr_line and is_cr:
            # CR→CR: заменяем предыдущий прогресс-бар
            self._log.delete("end-1c linestart", "end-1c")
        elif self._cr_line and not is_cr:
            # CR→non-CR: бар остаётся, новая строка ниже
            self._log.insert(tk.END, "\n")
        self._log.insert(tk.END, text, tag)
        self._cr_line = is_cr
        self._log.see(tk.END)
        self._log.configure(state="disabled")

    def _stream_output(self, process):
        """Читает stdout процесса побайтово, обрабатывает \\r для live progress."""
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        buf = ""
        while True:
            raw = process.stdout.read(1)
            if not raw:
                buf += decoder.decode(b'', final=True)
                break
            for ch in decoder.decode(raw):
                if ch == "\n":
                    self.after(0, self._log_line, buf + "\n", "stdout", False)
                    buf = ""
                elif ch == "\r":
                    if buf:
                        self.after(0, self._log_line, buf, "stdout", True)
                    buf = ""
                else:
                    buf += ch
        if buf:
            self.after(0, self._log_line, buf, "stdout", False)

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", tk.END)
        self._log.configure(state="disabled")

    def _copy_log(self):
        text = self._log.get("1.0", tk.END).strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self._show_toast("Log copied to clipboard")

    # ── Запуск процессов ──────────────────────────────────────────────────

    def _run_script(self, args, task_name):
        """Запускает скрипт в фоне, стримит вывод в лог."""
        if self._process and self._process.poll() is None:
            self._log_msg(f"[BUSY] Дождитесь завершения: {self._running_task}\n", "warn")
            return

        self._clear_log()
        self._log_msg(f"[START] {task_name}\n", "info")
        self._log_msg(f"$ {' '.join(args)}\n\n", "stdout")
        self._task_label.configure(text=f"Running: {task_name}", fg=FG_ACCENT)
        self._btn_stop.configure(state="normal")
        self._running_task = task_name

        def _worker():
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUNBUFFERED"] = "1"

                self._process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=BASE_DIR,
                    env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW
                        if sys.platform == "win32" else 0
                )

                self._stream_output(self._process)

                self._process.wait()
                code = self._process.returncode
                tag = "success" if code == 0 else "stderr"
                msg = "OK" if code == 0 else f"ERROR (code {code})"
                self.after(0, self._log_msg, f"\n[DONE] {task_name}: {msg}\n", tag)

            except Exception as e:
                self.after(0, self._log_msg, f"\n[ERROR] {e}\n", "stderr")

            finally:
                self._process = None
                self.after(0, self._on_task_done)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_task_done(self):
        task = self._running_task or "Task"
        self._task_label.configure(text="Idle", fg=FG_DIM)
        self._btn_stop.configure(state="disabled")
        self._running_task = None
        self._refresh_status()
        self._show_toast(f"{task} completed")

    # ── Toast-уведомления ─────────────────────────────────────────────────

    def _show_toast(self, message, duration=3000):
        """Всплывающее уведомление в правом нижнем углу."""
        toast = tk.Toplevel(self)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        toast.configure(bg=BG_CARD)

        tk.Label(
            toast, text=f"  {message}  ", font=("Consolas", 10),
            fg=FG_GREEN, bg=BG_CARD, padx=16, pady=8
        ).pack()

        # Позиция: правый нижний угол экрана
        toast.update_idletasks()
        w = toast.winfo_width()
        h = toast.winfo_height()
        sx = self.winfo_screenwidth()
        sy = self.winfo_screenheight()
        toast.geometry(f"+{sx - w - 20}+{sy - h - 60}")

        toast.after(duration, toast.destroy)

    def _stop_process(self):
        if self._process and self._process.poll() is None:
            self._log_msg("\n[STOP] Завершение процесса...\n", "warn")
            self._process.terminate()

    # ── Команды кнопок ────────────────────────────────────────────────────

    def _run_full(self):
        # Запускаем цепочку: data → train → dashboard
        self._run_chain([
            ([PY, os.path.join(BASE_DIR, "data_engine.py")], "Data Update"),
            ([PY, os.path.join(BASE_DIR, "train_hybrid.py")], "Train Models"),
            ([PY, "-m", "streamlit", "run", os.path.join(BASE_DIR, "app.py")], "Dashboard"),
        ])

    def _run_chain(self, steps):
        """Запускает несколько скриптов последовательно."""
        if self._process and self._process.poll() is None:
            self._log_msg(f"[BUSY] Дождитесь завершения: {self._running_task}\n", "warn")
            return

        self._clear_log()
        chain_name = " → ".join(s[1] for s in steps)
        self._log_msg(f"[CHAIN] {chain_name}\n\n", "info")
        self._task_label.configure(text="Running: Full Cycle", fg=FG_ACCENT)
        self._btn_stop.configure(state="normal")
        self._running_task = "Full Cycle"

        def _worker():
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUNBUFFERED"] = "1"

                for args, name in steps:
                    self.after(0, self._log_msg, f"\n{'='*50}\n[STEP] {name}\n{'='*50}\n", "info")

                    self._process = subprocess.Popen(
                        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        cwd=BASE_DIR, env=env,
                        creationflags=subprocess.CREATE_NO_WINDOW
                            if sys.platform == "win32" else 0
                    )

                    self._stream_output(self._process)

                    self._process.wait()
                    if self._process.returncode != 0:
                        self.after(0, self._log_msg,
                                   f"\n[FAIL] {name} (code {self._process.returncode})\n", "stderr")
                        break
                    self.after(0, self._log_msg, f"[OK] {name}\n", "success")
                else:
                    self.after(0, self._log_msg, "\n[DONE] Full Cycle complete!\n", "success")

            except Exception as e:
                self.after(0, self._log_msg, f"\n[ERROR] {e}\n", "stderr")
            finally:
                self._process = None
                self.after(0, self._on_task_done)

        threading.Thread(target=_worker, daemon=True).start()

    def _run_dashboard(self):
        self._run_script(
            [PY, "-m", "streamlit", "run", os.path.join(BASE_DIR, "app.py")],
            "Dashboard"
        )

    def _run_predict(self):
        self._run_script([PY, os.path.join(BASE_DIR, "predict.py")], "Predict")

    def _run_data(self):
        self._run_script([PY, os.path.join(BASE_DIR, "data_engine.py")], "Data Update")

    def _run_train(self):
        self._run_script([PY, os.path.join(BASE_DIR, "train_hybrid.py")], "Train Models")

    def _run_backtest(self):
        self._run_script([PY, os.path.join(BASE_DIR, "backtest.py")], "Backtest")

    def _run_bot(self):
        self._run_script([PY, os.path.join(BASE_DIR, "alert_bot.py")], "Telegram Bot")

    def _run_news(self):
        self._run_script([PY, os.path.join(BASE_DIR, "news_analyzer.py")], "News Analyzer")

    def _run_digest(self):
        self._run_script([PY, os.path.join(BASE_DIR, "news_analyzer.py"), "--digest"], "News Digest")

    def _run_regime(self):
        self._run_script([PY, os.path.join(BASE_DIR, "regime_detector.py")], "Regime Detector")

    def _run_corr(self):
        self._run_script([PY, os.path.join(BASE_DIR, "correlation_alert.py")], "Correlation Alert")

    def _run_paper(self):
        self._run_script([PY, os.path.join(BASE_DIR, "paper_trading.py"), "--status"], "Paper Trading")

    def _run_watchlist(self):
        self._open_watchlist_dialog()

    def _open_watchlist_dialog(self):
        """Интерактивное окно управления Watchlist."""
        dlg = tk.Toplevel(self)
        dlg.title("Watchlist Manager")
        dlg.configure(bg=BG)
        dlg.geometry("420x480")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        # ── Заголовок ──
        tk.Label(
            dlg, text="WATCHLIST", font=("Consolas", 14, "bold"),
            fg=FG_ACCENT, bg=BG
        ).pack(pady=(12, 4))

        # ── Выбор списка ──
        list_frame = tk.Frame(dlg, bg=BG)
        list_frame.pack(fill=tk.X, padx=16, pady=(8, 4))

        tk.Label(
            list_frame, text="List:", font=("Consolas", 10),
            fg=FG, bg=BG, width=8, anchor="w"
        ).pack(side=tk.LEFT)

        lists = ["default", "crypto", "us_tech", "russia", "macro"]
        # Load custom lists from watchlist.json
        try:
            import json as _json
            wl_path = os.path.join(BASE_DIR, "watchlist.json")
            if os.path.exists(wl_path):
                with open(wl_path, "r", encoding="utf-8") as _f:
                    custom = list(_json.load(_f).keys())
                lists = list(dict.fromkeys(custom + lists))
        except Exception:
            pass

        list_var = tk.StringVar(value="default")
        list_combo = ttk.Combobox(
            list_frame, textvariable=list_var, values=lists,
            state="readonly", width=20, font=("Consolas", 10)
        )
        list_combo.pack(side=tk.LEFT, padx=(8, 0))

        def _show_list():
            name = list_var.get()
            dlg.destroy()
            self._run_script(
                [PY, os.path.join(BASE_DIR, "watchlist.py"), "--list", name],
                f"Watchlist ({name})"
            )

        show_btn = tk.Button(
            list_frame, text="Show", font=("Consolas", 10, "bold"),
            fg=FG, bg=BG_BUTTON_ACTIVE, activeforeground=FG,
            activebackground=BG_BUTTON_HOVER, relief="flat",
            cursor="hand2", padx=10, command=_show_list
        )
        show_btn.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Separator(dlg, orient="horizontal").pack(fill=tk.X, padx=16, pady=12)

        # ── Добавить актив ──
        add_frame = tk.Frame(dlg, bg=BG)
        add_frame.pack(fill=tk.X, padx=16, pady=4)

        tk.Label(
            add_frame, text="Add:", font=("Consolas", 10),
            fg=FG, bg=BG, width=8, anchor="w"
        ).pack(side=tk.LEFT)

        add_entry = tk.Entry(
            add_frame, font=("Consolas", 10), bg=BG_CARD, fg=FG,
            insertbackground=FG, width=20, relief="flat",
            highlightbackground=BORDER, highlightthickness=1
        )
        add_entry.pack(side=tk.LEFT, padx=(8, 0))

        def _add_asset():
            assets = add_entry.get().strip().upper().split()
            name = list_var.get()
            if assets:
                dlg.destroy()
                self._run_script(
                    [PY, os.path.join(BASE_DIR, "watchlist.py"),
                     "--add"] + assets + ["--list", name],
                    f"Watchlist +{' '.join(assets)}"
                )

        tk.Button(
            add_frame, text="Add", font=("Consolas", 10),
            fg=FG_GREEN, bg=BG_BUTTON, activeforeground=FG,
            activebackground=BG_BUTTON_HOVER, relief="flat",
            cursor="hand2", padx=10, command=_add_asset
        ).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(
            dlg, text="  e.g.: TSLA DOGE   (space-separated)",
            font=("Consolas", 8), fg=FG_DIM, bg=BG, anchor="w"
        ).pack(fill=tk.X, padx=24)

        # ── Удалить актив ──
        rem_frame = tk.Frame(dlg, bg=BG)
        rem_frame.pack(fill=tk.X, padx=16, pady=(8, 4))

        tk.Label(
            rem_frame, text="Remove:", font=("Consolas", 10),
            fg=FG, bg=BG, width=8, anchor="w"
        ).pack(side=tk.LEFT)

        rem_entry = tk.Entry(
            rem_frame, font=("Consolas", 10), bg=BG_CARD, fg=FG,
            insertbackground=FG, width=20, relief="flat",
            highlightbackground=BORDER, highlightthickness=1
        )
        rem_entry.pack(side=tk.LEFT, padx=(8, 0))

        def _remove_asset():
            assets = rem_entry.get().strip().upper().split()
            name = list_var.get()
            if assets:
                dlg.destroy()
                self._run_script(
                    [PY, os.path.join(BASE_DIR, "watchlist.py"),
                     "--remove"] + assets + ["--list", name],
                    f"Watchlist -{' '.join(assets)}"
                )

        tk.Button(
            rem_frame, text="Del", font=("Consolas", 10),
            fg=FG_RED, bg=BG_BUTTON, activeforeground=FG,
            activebackground=BG_BUTTON_HOVER, relief="flat",
            cursor="hand2", padx=10, command=_remove_asset
        ).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Separator(dlg, orient="horizontal").pack(fill=tk.X, padx=16, pady=12)

        # ── Создать новый список ──
        tk.Label(
            dlg, text="CREATE NEW LIST", font=("Consolas", 10, "bold"),
            fg=FG_DIM, bg=BG, anchor="w"
        ).pack(fill=tk.X, padx=16)

        new_frame = tk.Frame(dlg, bg=BG)
        new_frame.pack(fill=tk.X, padx=16, pady=(4, 4))

        tk.Label(
            new_frame, text="Name:", font=("Consolas", 10),
            fg=FG, bg=BG, width=8, anchor="w"
        ).pack(side=tk.LEFT)

        new_name_entry = tk.Entry(
            new_frame, font=("Consolas", 10), bg=BG_CARD, fg=FG,
            insertbackground=FG, width=20, relief="flat",
            highlightbackground=BORDER, highlightthickness=1
        )
        new_name_entry.pack(side=tk.LEFT, padx=(8, 0))

        assets_frame = tk.Frame(dlg, bg=BG)
        assets_frame.pack(fill=tk.X, padx=16, pady=(4, 4))

        tk.Label(
            assets_frame, text="Assets:", font=("Consolas", 10),
            fg=FG, bg=BG, width=8, anchor="w"
        ).pack(side=tk.LEFT)

        new_assets_entry = tk.Entry(
            assets_frame, font=("Consolas", 10), bg=BG_CARD, fg=FG,
            insertbackground=FG, width=20, relief="flat",
            highlightbackground=BORDER, highlightthickness=1
        )
        new_assets_entry.pack(side=tk.LEFT, padx=(8, 0))

        def _create_list():
            name = new_name_entry.get().strip().lower()
            assets = new_assets_entry.get().strip().upper().split()
            if name and assets:
                dlg.destroy()
                self._run_script(
                    [PY, os.path.join(BASE_DIR, "watchlist.py"),
                     "--create", name] + assets,
                    f"Watchlist create '{name}'"
                )

        tk.Button(
            assets_frame, text="Create", font=("Consolas", 10),
            fg=FG_ACCENT, bg=BG_BUTTON, activeforeground=FG,
            activebackground=BG_BUTTON_HOVER, relief="flat",
            cursor="hand2", padx=10, command=_create_list
        ).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(
            dlg, text="  e.g.: Name: forex  Assets: EURUSD GBPUSD USDJPY",
            font=("Consolas", 8), fg=FG_DIM, bg=BG, anchor="w"
        ).pack(fill=tk.X, padx=24, pady=(0, 4))

        # ── Показать все списки ──
        ttk.Separator(dlg, orient="horizontal").pack(fill=tk.X, padx=16, pady=8)

        def _show_all():
            dlg.destroy()
            self._run_script(
                [PY, os.path.join(BASE_DIR, "watchlist.py"), "--lists"],
                "Watchlist (all lists)"
            )

        tk.Button(
            dlg, text="Show All Lists", font=("Consolas", 10),
            fg=FG, bg=BG_BUTTON, activeforeground=FG,
            activebackground=BG_BUTTON_HOVER, relief="flat",
            cursor="hand2", padx=12, pady=6, command=_show_all
        ).pack(pady=(0, 12))

    def _run_optuna(self):
        self._run_script([PY, os.path.join(BASE_DIR, "optuna_tune.py")], "Optuna Tune")

    def _run_signal_radar(self):
        self._run_script([PY, os.path.join(BASE_DIR, "signal_dashboard.py")], "Signal Radar")

    def _run_sector_rotation(self):
        self._run_script([PY, os.path.join(BASE_DIR, "sector_rotation.py")], "Sector Rotation")

    def _run_whatif_top5(self):
        self._run_script([PY, os.path.join(BASE_DIR, "whatif_simulator.py"),
                          "--top", "5", "--days", "90"], "What-If Top-5")

    def _run_whatif_top10(self):
        self._run_script([PY, os.path.join(BASE_DIR, "whatif_simulator.py"),
                          "--top", "10", "--days", "90"], "What-If Top-10")

    def _run_whatif_180(self):
        self._run_script([PY, os.path.join(BASE_DIR, "whatif_simulator.py"),
                          "--top", "5", "--days", "180"], "What-If 180 Days")

    def _run_whatif_kelly(self):
        self._run_script([PY, os.path.join(BASE_DIR, "whatif_simulator.py"),
                          "--top", "5", "--days", "90", "--strategy", "kelly"],
                         "What-If Kelly")

    def _run_whatif_custom(self):
        import tkinter.simpledialog as sd
        assets_str = sd.askstring(
            "What-If Custom",
            "Активы через пробел (например: BTC ETH NVDA GOLD):",
            parent=self.root,
        )
        if not assets_str:
            return
        days_str = sd.askstring(
            "What-If Custom",
            "Количество дней (по умолчанию 90):",
            parent=self.root,
        )
        days = days_str.strip() if days_str and days_str.strip().isdigit() else "90"
        assets = assets_str.upper().split()
        self._run_script(
            [PY, os.path.join(BASE_DIR, "whatif_simulator.py")] + assets +
            ["--days", days],
            "What-If Custom",
        )

    def _run_auto_trader(self):
        self._run_script([PY, os.path.join(BASE_DIR, "auto_trader.py"), "--dry-run"], "Auto-Trader (dry)")

    def _run_check_alerts(self):
        self._run_script([PY, os.path.join(BASE_DIR, "alert_rules.py")], "Check Alerts")

    def _run_model_compare(self):
        self._run_script([PY, os.path.join(BASE_DIR, "model_comparison.py")], "Model Compare")

    def _run_performance(self):
        self._run_script([PY, os.path.join(BASE_DIR, "performance_tracker.py")], "Performance")

    def _run_guru_track(self):
        self._run_script([PY, os.path.join(BASE_DIR, "guru_tracker.py")], "Guru Track")

    def _run_guru_report(self):
        self._open_guru_report_dialog()

    def _open_guru_report_dialog(self):
        """Dialog to select assets for Guru Council report."""
        dlg = tk.Toplevel(self)
        dlg.title("Guru Council Report")
        dlg.configure(bg=BG)
        dlg.geometry("420x350")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        tk.Label(
            dlg, text="GURU REPORT", font=("Consolas", 14, "bold"),
            fg=FG_ACCENT, bg=BG
        ).pack(pady=(12, 4))

        tk.Label(
            dlg, text="Фундаментальная отчётность + вердикт гуру",
            font=("Consolas", 9), fg=FG_DIM, bg=BG
        ).pack(pady=(0, 8))

        # Quick buttons
        btn_frame = tk.Frame(dlg, bg=BG)
        btn_frame.pack(fill=tk.X, padx=16, pady=4)

        def _run_all_short():
            dlg.destroy()
            self._run_script(
                [PY, os.path.join(BASE_DIR, "guru_report.py")],
                "Guru Report (all)"
            )

        def _run_all_full():
            dlg.destroy()
            self._run_script(
                [PY, os.path.join(BASE_DIR, "guru_report.py"), "--all"],
                "Guru Report (all full)"
            )

        tk.Button(
            btn_frame, text="All (summary)", font=("Consolas", 10),
            fg=FG, bg=BG_BUTTON, activeforeground=FG, activebackground=BG_BUTTON_ACTIVE,
            relief="flat", cursor="hand2", padx=10, pady=6, command=_run_all_short
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            btn_frame, text="All (full detail)", font=("Consolas", 10),
            fg=FG, bg=BG_BUTTON, activeforeground=FG, activebackground=BG_BUTTON_ACTIVE,
            relief="flat", cursor="hand2", padx=10, pady=6, command=_run_all_full
        ).pack(fill=tk.X, pady=2)

        ttk.Separator(dlg, orient="horizontal").pack(fill=tk.X, padx=16, pady=8)

        # Sector buttons
        tk.Label(
            dlg, text="BY SECTOR", font=("Consolas", 9, "bold"),
            fg=FG_DIM, bg=BG, anchor="w"
        ).pack(fill=tk.X, padx=16)

        sector_frame = tk.Frame(dlg, bg=BG)
        sector_frame.pack(fill=tk.X, padx=16, pady=4)

        for sec_name, sec_label in [("US", "US Tech"), ("RUS", "Russia"),
                                     ("CRYPTO", "Crypto"), ("COMMODITY", "Commodities")]:
            def _run_sector(s=sec_name):
                dlg.destroy()
                self._run_script(
                    [PY, os.path.join(BASE_DIR, "guru_report.py"), "--sector", s],
                    f"Guru Report ({s})"
                )
            tk.Button(
                sector_frame, text=sec_label, font=("Consolas", 9),
                fg=FG, bg=BG_BUTTON, activeforeground=FG, activebackground=BG_BUTTON_HOVER,
                relief="flat", cursor="hand2", padx=8, pady=4, command=_run_sector
            ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(dlg, orient="horizontal").pack(fill=tk.X, padx=16, pady=8)

        # Custom asset input
        custom_frame = tk.Frame(dlg, bg=BG)
        custom_frame.pack(fill=tk.X, padx=16, pady=4)

        tk.Label(
            custom_frame, text="Asset:", font=("Consolas", 10),
            fg=FG, bg=BG, width=7, anchor="w"
        ).pack(side=tk.LEFT)

        asset_entry = tk.Entry(
            custom_frame, font=("Consolas", 10), bg=BG_CARD, fg=FG,
            insertbackground=FG, width=20, relief="flat",
            highlightbackground=BORDER, highlightthickness=1
        )
        asset_entry.pack(side=tk.LEFT, padx=(8, 0))

        def _run_custom():
            text = asset_entry.get().strip().upper()
            if text:
                dlg.destroy()
                assets = text.split()
                self._run_script(
                    [PY, os.path.join(BASE_DIR, "guru_report.py")] + assets,
                    f"Guru Report ({' '.join(assets[:3])})"
                )

        tk.Button(
            custom_frame, text="Go", font=("Consolas", 10, "bold"),
            fg=FG, bg=BG_BUTTON_ACTIVE, activeforeground=FG,
            activebackground=BG_BUTTON_HOVER, relief="flat",
            cursor="hand2", padx=10, command=_run_custom
        ).pack(side=tk.LEFT, padx=(8, 0))

        asset_entry.bind("<Return>", lambda e: _run_custom())

        tk.Label(
            dlg, text="  e.g.: TSLA NVDA SBER   (space-separated)",
            font=("Consolas", 8), fg=FG_DIM, bg=BG, anchor="w"
        ).pack(fill=tk.X, padx=24, pady=(2, 8))

    def _run_model_health(self):
        self._run_script([PY, os.path.join(BASE_DIR, "model_health.py")], "Model Health")

    def _run_export(self):
        self._run_script([PY, os.path.join(BASE_DIR, "export_signals.py")], "Export Signals")

    def _run_signal_log(self):
        self._run_script([PY, os.path.join(BASE_DIR, "signal_log.py")], "Signal Log")

    def _run_report(self):
        self._run_script([PY, os.path.join(BASE_DIR, "performance_report.py")], "HTML Report")

    def _run_equity(self):
        self._run_script([PY, os.path.join(BASE_DIR, "equity_curve.py")], "Equity Curve")

    def _run_scheduler(self):
        self._run_script([PY, os.path.join(BASE_DIR, "scheduler.py")], "Scheduler")

    def _run_backup(self):
        self._run_script([PY, os.path.join(BASE_DIR, "db_backup.py"), "--auto"], "DB Backup")

    def _run_db_check(self):
        self._run_script([PY, os.path.join(BASE_DIR, "db_check.py")], "DB Check")

    def _run_db_fix(self):
        self._run_script([PY, os.path.join(BASE_DIR, "db_check.py"), "--fix"], "DB Fix")

    def _run_install(self):
        self._run_script(
            [PY, "-m", "pip", "install",
             "apimoex", "requests", "yfinance", "pandas", "numpy<2",
             "tensorflow==2.10.0", "protobuf>=3.20,<4", "plotly", "streamlit",
             "sqlalchemy", "catboost", "scikit-learn", "pyTelegramBotAPI",
             "pysocks", "python-dotenv", "tabulate", "tqdm",
             "--no-cache-dir"],
            "Install/Repair"
        )

    # ── Закрытие ──────────────────────────────────────────────────────────

    def destroy(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        super().destroy()


if __name__ == "__main__":
    app = GTradeLauncher()
    app.mainloop()
