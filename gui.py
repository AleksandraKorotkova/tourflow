"""Интерфейс TourFlow. Запуск: python gui.py."""
from __future__ import annotations

import csv
import json
import os
import queue
import shutil
import sqlite3
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

import yaml
from PIL import Image, ImageTk

import gui_tasks as tasks
import gui_theme as theme
from gui_accounts import ADMIN, ANALYST, ROLES, Account, Accounts

CHARTS = {
    "08_forecast": "Факт и прогноз с интервалом",
    "01_series": "Динамика и сезонная модель",
    "02_seasonality": "Сезонный индекс по месяцам",
    "03_residuals": "Остатки и аномалии",
    "04_lag_fx": "Связь с курсом доллара",
    "05_models_mape": "Сравнение моделей по MAPE",
    "06_error_horizon": "Ошибка по шагам прогноза",
    "07_backtest": "Исторические прогнозы",
    "09_factor_effects": "Коэффициенты регрессии",
}
LOAD_STATUS = {
    "success": "Успешно", "failed": "Ошибка", "skipped": "Пропущено",
    "running": "Выполняется",
}
USER_ERRORS = (ValueError, PermissionError, sqlite3.Error, OSError)


def setup_style(root):
    available = {family.casefold() for family in tkfont.families(root=root)}
    family = next((name for name in (theme.FONT_FAMILY, "DejaVu Sans", "Arial")
                   if name.casefold() in available),
                  tkfont.nametofont("TkDefaultFont", root=root).actual("family"))
    root.option_add("*Text.font", (family, theme.FONT_SIZE))
    root.option_add("*TCombobox*Listbox.font", (family, theme.FONT_SIZE))
    root.option_add("*TCombobox*Listbox.foreground", theme.TEXT)
    root.configure(background=theme.BACKGROUND)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=(family, theme.FONT_SIZE),
                    background=theme.SURFACE, foreground=theme.TEXT)
    style.configure("Shell.TFrame", background=theme.BACKGROUND)
    style.configure("Shell.TLabel", background=theme.BACKGROUND)
    style.configure("Role.TLabel", background=theme.BACKGROUND,
                    foreground=theme.SECONDARY_TEXT)
    style.configure("Title.TLabel", font=(family, 21, "bold"))
    style.configure("Page.TLabel", font=(family, 15, "bold"))
    style.configure("Brand.TLabel", font=(family, 20, "bold"),
                    background=theme.BACKGROUND)
    style.configure("Hint.TLabel", foreground=theme.SECONDARY_TEXT)
    style.configure("Error.TLabel", foreground=theme.ERROR)
    style.configure("Success.TLabel", foreground=theme.SUCCESS)
    style.configure("TButton", padding=(11, 6), bordercolor=theme.BORDER)
    style.configure("Primary.TButton", background=theme.ACCENT, foreground=theme.ON_ACCENT,
                    bordercolor=theme.ACCENT)
    style.map("Primary.TButton",
              background=[("disabled", theme.TABLE_ALTERNATE), ("active", theme.ACCENT_HOVER)],
              foreground=[("disabled", theme.SECONDARY_TEXT)])
    style.configure("TNotebook", background=theme.BACKGROUND, bordercolor=theme.BORDER)
    style.configure("TNotebook.Tab", padding=(15, 9), background=theme.TABLE_ALTERNATE,
                    foreground=theme.TEXT)
    style.map("TNotebook.Tab", background=[("selected", theme.ACCENT)],
              foreground=[("selected", theme.ON_ACCENT)], padding=[("selected", (15, 9))])
    def without_tab_focus(elements):
        result = []
        for element, options in elements:
            options = options.copy()
            if "children" in options:
                options["children"] = without_tab_focus(options["children"])
            if element.lower().endswith(".focus"):
                result.extend(options.get("children", []))
            else:
                result.append((element, options))
        return result

    style.layout(
        "TNotebook.Tab",
        without_tab_focus(style.layout("TNotebook.Tab")),
    )
    style.configure("Treeview", rowheight=27, fieldbackground=theme.SURFACE,
                    background=theme.SURFACE, foreground=theme.TEXT)
    style.configure("Treeview.Heading", font=(family, theme.FONT_SIZE, "bold"),
                    background=theme.TABLE_ALTERNATE, foreground=theme.TEXT)
    style.map("Treeview", background=[("selected", theme.ACCENT)],
              foreground=[("selected", theme.ON_ACCENT)])
    style.map("TCombobox", fieldbackground=[("readonly", theme.SURFACE)],
              foreground=[("readonly", theme.TEXT)])
    style.configure("Horizontal.TProgressbar", background=theme.ACCENT, troughcolor=theme.BACKGROUND)


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def format_number(value, decimals=0):
    try:
        number = float(value)
        if number != number or abs(number) == float("inf"):
            return "—"
        return f"{number:,.{decimals}f}".replace(",", " ").replace(".", ",")
    except (ValueError, TypeError):
        return "—"


def month(value):
    year, sep, number = value.partition("-")
    return f"{number}.{year}" if sep else value


def local_time(value):
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%d.%m.%Y %H:%M")
    except (ValueError, TypeError):
        return value or "—"


def table(parent, headings, height, widths=None):
    frame = ttk.Frame(parent)
    columns = list(range(len(headings)))
    view = ttk.Treeview(frame, columns=columns, show="headings", height=height)
    scroll = ttk.Scrollbar(frame, orient="vertical", command=view.yview)
    view.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    view.pack(side="left", fill="both", expand=True)
    for index, title in enumerate(headings):
        view.heading(index, text=title)
        view.column(index, width=widths[index] if widths else 135,
                    minwidth=65, anchor="w" if index == 0 else "center")
    view.tag_configure("alternate", background=theme.TABLE_ALTERNATE)
    return frame, view


def set_rows(view, rows):
    view.delete(*view.get_children())
    for index, row in enumerate(rows):
        view.insert("", "end", values=row, tags=("alternate",) if index % 2 else ())


def text_panel(parent, height):
    frame = ttk.Frame(parent)
    view = tk.Text(frame, height=height, wrap="word", relief="solid", borderwidth=0,
                   highlightthickness=1, highlightbackground=theme.BORDER,
                   background=theme.SURFACE, foreground=theme.TEXT, padx=11, pady=9,
                   insertbackground=theme.TEXT, spacing3=5)
    scroll = ttk.Scrollbar(frame, orient="vertical", command=view.yview)
    view.configure(yscrollcommand=scroll.set, state="disabled")
    scroll.pack(side="right", fill="y")
    view.pack(side="left", fill="both", expand=True)
    return frame, view


def set_text(view, value, append=False):
    view.configure(state="normal")
    if not append:
        view.delete("1.0", "end")
    view.insert("end", value)
    if append:
        view.see("end")
    view.configure(state="disabled")


class TourFlow:
    def __init__(self, root):
        self.root = root
        self.accounts = Accounts(tasks.STATE / "accounts.db")
        self.account: Account | None = None
        self.events = queue.Queue()
        self.running = False
        self.reading_journal = False
        self.results = tasks.active_results()
        self.chart_path = None
        self.chart_source = None
        self.chart_photo = None
        self.chart_callback = None
        self.journal_rows = {}
        root.title("TourFlow")
        max_width, max_height = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1160, max_width - 32)}x{min(830, max_height - 82)}")
        root.minsize(940, 640)
        setup_style(root)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.show_login()
        root.after(70, self.poll)

    def clear(self):
        if self.chart_callback:
            self.root.after_cancel(self.chart_callback)
            self.chart_callback = None
        if self.chart_source:
            self.chart_source.close()
            self.chart_source = None
        self.chart_photo = None
        for child in self.root.winfo_children():
            child.destroy()

    def show_login(self):
        self.clear()
        self.account = None
        initial = self.accounts.needs_setup()
        card = ttk.Frame(self.root, padding=30, borderwidth=1, relief="solid")
        card.place(relx=0.5, rely=0.46, anchor="center")
        ttk.Label(card, text="TourFlow", style="Title.TLabel").grid(sticky="w")
        title = "Создание администратора" if initial else "Вход в приложение"
        ttk.Label(card, text=title, style="Page.TLabel").grid(sticky="w", pady=(4, 12))
        values = []
        fields = [("Логин", False), ("Пароль", True)]
        if initial:
            fields.append(("Повторите пароль", True))
        for title, secret in fields:
            ttk.Label(card, text=title).grid(sticky="w")
            entry = ttk.Entry(card, width=36, show="•" if secret else "")
            entry.grid(sticky="ew", pady=(3, 9))
            values.append(entry)
        if initial:
            ttk.Label(card, text="Пароль — минимум 8 символов.",
                      style="Hint.TLabel").grid(sticky="w")
        error = ttk.Label(card, style="Error.TLabel", wraplength=340)
        error.grid(sticky="w", pady=(4, 8))

        def submit(_event=None):
            login, password = (entry.get() for entry in values[:2])
            try:
                if initial:
                    if password != values[2].get():
                        raise ValueError("Пароли не совпадают.")
                    account = self.accounts.create(login, password, ADMIN, None)
                else:
                    account = self.accounts.authenticate(login, password)
            except USER_ERRORS as exc:
                error.configure(text=str(exc))
                for field in values[1:]:
                    field.delete(0, "end")
                values[1].focus_set()
                return
            self.account = account
            self.show_workspace()

        ttk.Button(card, text="Создать и войти" if initial else "Войти",
                   command=submit, style="Primary.TButton").grid(sticky="ew")
        for field in values:
            field.bind("<Return>", submit)
        values[0].focus_set()

    def show_workspace(self):
        self.clear()
        shell = ttk.Frame(self.root, padding=(18, 12), style="Shell.TFrame")
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell, style="Shell.TFrame")
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="TourFlow. Турпоток в Краснодарском Крае", style="Brand.TLabel").pack(side="left")
        self.logout = ttk.Button(header, text="Выйти", command=self.sign_out)
        self.logout.pack(side="right")
        if self.account.role == ADMIN:
            ttk.Button(header, text="Учётные записи", command=self.manage_accounts).pack(
                side="right", padx=(8, 10)
            )
        ttk.Label(header, text=f"{self.account.login} · {ROLES[self.account.role]}",
                  style="Role.TLabel").pack(side="right", padx=8)
        self.tabs = ttk.Notebook(shell)
        self.tabs.pack(fill="both", expand=True)
        footer = ttk.Frame(shell, style="Shell.TFrame")
        footer.pack(fill="x", pady=(8, 0))
        self.status = ttk.Label(footer, style="Shell.TLabel", text="Готово к работе")
        self.status.pack(side="left")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=155)
        self.build_forecast(self.page("Прогноз"))
        self.build_quality(self.page("Качество моделей"))
        panel, self.conclusions = text_panel(self.page("Выводы"), 18)
        panel.pack(fill="both", expand=True)
        if self.account.role == ADMIN:
            self.build_load(self.page("Загрузка данных"))
            self.journal_tab = self.page("Журнал")
            self.build_journal(self.journal_tab)
        self.refresh_results()

    def page(self, name):
        page = ttk.Frame(self.tabs, padding=(14, 12))
        self.tabs.add(page, text=name)
        return page

    def build_forecast(self, page):
        self.summary = ttk.Label(page, style="Page.TLabel")
        self.summary.pack(anchor="w")
        self.date_label = ttk.Label(page, style="Hint.TLabel")
        self.date_label.pack(anchor="w", pady=(3, 7))
        actions = ttk.Frame(page)
        actions.pack(fill="x", pady=(0, 6))
        ttk.Label(actions, text="График").pack(side="left", padx=(0, 6))
        self.chart_select = ttk.Combobox(actions, state="readonly", width=28)
        self.chart_select.pack(side="left", fill="x", expand=True)
        self.chart_select.bind("<<ComboboxSelected>>", self.choose_chart)
        self.save_png = ttk.Button(actions, text="Сохранить PNG", command=self.export_png)
        self.save_png.pack(side="left", padx=(8, 5))
        ttk.Button(actions, text="Сохранить CSV",
                   command=lambda: self.export(self.results / "forecast.csv")).pack(side="left")
        self.analyze_button = ttk.Button(
            actions, text="Пересчитать по готовым данным", style="Primary.TButton",
            command=lambda: self.start_calculation("analyze"),
        )
        self.analyze_button.pack(side="right", padx=(8, 0))
        ttk.Button(
            page,
            text="Открыть папку результатов",
            command=self.open_results,
        ).pack(side="bottom", anchor="w", pady=(8, 0))

        content = ttk.Frame(page)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=2, uniform="forecast")
        content.columnconfigure(1, weight=1, uniform="forecast")
        content.rowconfigure(0, weight=1)

        chart_panel = ttk.Frame(content)
        chart_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        self.legend = ttk.Label(chart_panel, style="Hint.TLabel", wraplength=500)
        self.legend.pack(side="bottom", anchor="w", pady=(6, 0))
        chart_panel.bind(
            "<Configure>",
            lambda event: self.legend.configure(
                wraplength=max(1, event.width - 12)
            ),
        )

        self.chart_label = tk.Canvas(
            chart_panel,
            width=1,
            height=1,
            background=theme.SURFACE,
            highlightthickness=1,
            highlightbackground=theme.BORDER,
        )
        self.chart_label.pack(fill="both", expand=True)
        self.chart_label.bind("<Configure>", self.schedule_chart)

        table_panel = ttk.Frame(content)
        table_panel.grid(row=0, column=1, sticky="new")

        ttk.Label(
            table_panel,
            text="Прогноз по месяцам, чел.",
            style="Page.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        headings = (
            "Месяц",
            "Прогноз,\nчел.",
            "Нижняя\nграница",
            "Верхняя\nграница",
        )
        frame, self.forecast_table = table(
            table_panel,
            headings,
            height=6,
            widths=(70, 90, 80, 80),
        )

        view = self.forecast_table
        view.configure(show="")

        header = ttk.Frame(frame)
        header.pack(side="top", fill="x", before=view)

        heading_font = ttk.Style(view).lookup("Treeview.Heading", "font")
        labels = [
            tk.Label(
                header,
                text=title,
                font=heading_font,
                background=theme.TABLE_ALTERNATE,
                foreground=theme.TEXT,
                justify="center",
                relief="groove",
                borderwidth=1,
                pady=4,
            )
            for title in headings
        ]
        header.configure(height=max(label.winfo_reqheight() for label in labels))

        def align_headings():
            available = header.winfo_width()
            if available <= 1:
                return

            widths = [view.column(index, "width") for index in range(len(labels))]
            total = sum(widths)
            x = 0
            used = 0

            for label, width in zip(labels, widths):
                used += width
                right = round(available * used / total)
                label.place(x=x, y=0, width=right - x, relheight=1)
                x = right

        header.bind("<Configure>", lambda _event: header.after_idle(align_headings))
        frame.pack(fill="x")

    def build_quality(self, page):
        self.model_label = ttk.Label(page, style="Page.TLabel")
        self.model_label.pack(anchor="w")
        ttk.Label(page, text="MAPE — средняя абсолютная процентная ошибка. "
                  "RMSE — среднеквадратичная ошибка в людях.",
                  style="Hint.TLabel", wraplength=780).pack(anchor="w", pady=(5, 10))
        frame, self.metrics_table = table(
            page, ("Модель", "MAPE, %", "RMSE, чел.", "Пар «факт — прогноз»"), 4,
            (225, 110, 130, 175)
        )
        frame.pack(fill="x")
        ttk.Button(page, text="Сохранить метрики CSV",
                   command=lambda: self.export(self.results / "metrics.csv")).pack(
            anchor="w", pady=(6, 13)
        )
        ttk.Label(page, text="Историческое покрытие интервалов",
                  style="Page.TLabel").pack(anchor="w")
        self.coverage_label = ttk.Label(page, style="Hint.TLabel", wraplength=780)
        self.coverage_label.pack(anchor="w", pady=(5, 8))
        frame, self.coverage_table = table(
            page, ("Модель", "Шаг, мес.", "Проверок", "Средняя ширина, чел.", "Покрытие, %"), 8,
            (195, 95, 100, 165, 115)
        )
        frame.pack(fill="both", expand=True)

    def build_load(self, page):
        ttk.Label(page, text="Обновление данных", style="Page.TLabel").pack(anchor="w")
        ttk.Label(page, text="Файл Росстата", style="Hint.TLabel").pack(anchor="w", pady=(9, 4))
        line = ttk.Frame(page)
        line.pack(fill="x")
        try:
            default_file = str(tasks.rosstat_path())
        except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as exc:
            default_file = ""
            self.set_status(f"Проверьте настройку файла Росстата. {exc}")
        self.excel = tk.StringVar(value=default_file)
        ttk.Entry(line, textvariable=self.excel, state="readonly").pack(
            side="left", fill="x", expand=True
        )
        self.excel_button = ttk.Button(line, text="Выбрать Excel…", command=self.pick_excel)
        self.excel_button.pack(side="right", padx=(8, 0))
        ttk.Label(page, text="Последовательно выполняются загрузка источников, сборка витрины "
                  "PostgreSQL и CSV, анализ и прогноз.", style="Hint.TLabel",
                  wraplength=820).pack(anchor="w", pady=(8, 7))
        self.update_button = ttk.Button(page, text="Обновить данные и пересчитать",
                                        style="Primary.TButton",
                                        command=lambda: self.start_calculation("update"))
        self.update_button.pack(anchor="w")
        ttk.Label(page, text="Ход выполнения", style="Page.TLabel").pack(anchor="w", pady=(16, 6))
        panel, self.activity = text_panel(page, 14)
        panel.pack(fill="both", expand=True)
        set_text(self.activity, "Здесь появятся этапы и сообщения источников после запуска.")

    def build_journal(self, page):
        bar = ttk.Frame(page)
        bar.pack(fill="x", pady=(0, 8))
        ttk.Label(bar, text="Журнал загрузок PostgreSQL", style="Page.TLabel").pack(side="left")
        self.journal_button = ttk.Button(bar, text="Обновить журнал", command=self.request_journal)
        self.journal_button.pack(side="right")
        self.journal_status = ttk.Label(page, text="Нажмите «Обновить журнал», чтобы загрузить записи.",
                                        style="Hint.TLabel")
        self.journal_status.pack(anchor="w", pady=(0, 8))
        frame, self.journal_table = table(
            page, ("Источник", "Время запуска", "Статус", "Прочитано", "Записано", "Период"), 11,
            (120, 135, 155, 82, 82, 170)
        )
        frame.pack(fill="both", expand=True)
        self.journal_table.bind("<<TreeviewSelect>>", self.show_journal_detail)
        ttk.Label(page, text="Подробности выбранной записи").pack(anchor="w", pady=(9, 5))
        panel, self.journal_detail = text_panel(page, 4)
        panel.pack(fill="x")

    def refresh_results(self):
        self.results = tasks.active_results()
        meta_file = self.results / "run.json"
        try:
            metadata = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
            quality_file = self.results / "quality.json"
            quality = json.loads(quality_file.read_text(encoding="utf-8")) if quality_file.is_file() else {}
            forecasts = read_csv(self.results / "forecast.csv")
            metrics = read_csv(self.results / "metrics.csv")
            coverage = read_csv(self.results / "interval_coverage.csv")
            conclusion_file = self.results / "conclusions.txt"
            conclusions = conclusion_file.read_text(encoding="utf-8-sig") if conclusion_file.is_file() else ""
        except (OSError, ValueError, csv.Error) as exc:
            self.set_status(f"Не удалось прочитать результаты. {exc}")
            return

        last_fact = month((quality.get("period") or ["—"])[-1])
        if forecasts:
            first, last = month(forecasts[0]["period"]), month(forecasts[-1]["period"])
            period = first if first == last else f"{first} — {last}"
            self.summary.configure(text=f"Последний факт · {last_fact}     Период прогноза · {period}")
        else:
            self.summary.configure(text="Прогноз появится после расчёта")
        finished = metadata.get("finished_at")
        if finished:
            date_label = f"Расчёт выполнен {local_time(finished)}"
        elif forecasts:
            date_label = "Сохранённый пример · дата расчёта не записана"
        else:
            date_label = "Запустите пересчёт по подготовленной месячной витрине."
        if metadata.get("warnings"):
            date_label += " · предупреждения при загрузке источников"
        self.date_label.configure(text=date_label)
        level = metadata.get("interval_level", 0.9)
        self.forecast_legend = (
            f"Синий — факт. Оранжевый — прогноз. Полоса — интервал с номинальным уровнем {level:.0%}."
        )
        set_rows(self.forecast_table, [
            (month(row["period"]), *(format_number(row.get(key)) for key in ("forecast", "lower", "upper")))
            for row in forecasts
        ])
        model = metadata.get("best_model")
        if not model and metrics:
            model = min(metrics, key=lambda row: float(row["MAPE"]))["model"]
        selected = next((row for row in metrics if row["model"] == model), None)
        self.model_label.configure(
            text=f"{model} · MAPE {format_number(selected['MAPE'], 2)} %" if selected
            else "Метрики появятся после расчёта"
        )
        set_rows(self.metrics_table, [
            (row["model"], format_number(row.get("MAPE"), 2), format_number(row.get("RMSE")),
             row.get("n", "—")) for row in metrics
        ])
        self.coverage_label.configure(text=(
            "Доля известных фактов, попавших в интервалы при исторической проверке."
            if coverage else "В этом расчёте нет данных для проверки покрытия."
        ))
        set_rows(self.coverage_table, [
            (row["model"], row.get("h"), row.get("n"), format_number(row.get("mean_width")),
             format_number(row.get("coverage_pct"), 1)) for row in coverage
        ])
        set_text(self.conclusions, conclusions or "Выводы появятся после расчёта.")
        available = {path.stem: path for path in (self.results / "figures").glob("*.png")}
        self.charts = {label: available.pop(key) for key, label in CHARTS.items() if key in available}
        self.charts.update((name, path) for name, path in sorted(available.items()))
        current = self.chart_select.get()
        self.chart_select.configure(values=list(self.charts))
        self.chart_select.set(current if current in self.charts else next(iter(self.charts), ""))
        self.choose_chart()

    def choose_chart(self, _event=None):
        self.chart_path = self.charts.get(self.chart_select.get())
        if self.chart_source:
            self.chart_source.close()
        self.chart_source = None
        self.chart_photo = None
        self.chart_label.delete("all")
        self.chart_message = ""
        self.legend.configure(text=(
            self.forecast_legend if self.chart_select.get() == CHARTS["08_forecast"] else " "
        ))
        if self.chart_path:
            try:
                with Image.open(self.chart_path) as image:
                    self.chart_source = image.convert("RGB")
            except OSError as exc:
                self.chart_message = f"График не открывается. {exc}"
        else:
            self.chart_message = "Графики появятся после расчёта."
        self.save_png.configure(state="normal" if self.chart_source else "disabled")
        self.schedule_chart()

    def schedule_chart(self, _event=None):
        if self.chart_callback:
            self.root.after_cancel(self.chart_callback)
        self.chart_callback = self.root.after(65, self.draw_chart)

    def draw_chart(self):
        self.chart_callback = None
        width, height = self.chart_label.winfo_width(), self.chart_label.winfo_height()
        if width < 30 or height < 30:
            return
        self.chart_label.delete("all")
        if self.chart_source is None:
            self.chart_label.create_text(width / 2, height / 2, text=self.chart_message,
                                         fill=theme.SECONDARY_TEXT, width=width - 20)
            return
        image = self.chart_source.copy()
        image.thumbnail((width - 10, height - 10), Image.Resampling.LANCZOS)
        self.chart_photo = ImageTk.PhotoImage(image)
        self.chart_label.create_image(width / 2, height / 2, image=self.chart_photo)

    def set_status(self, text):
        self.status.configure(text=text.partition("\n")[0][:122])

    def export_png(self):
        if self.chart_path:
            self.export(self.chart_path)

    def export(self, source):
        if not source.is_file():
            messagebox.showinfo("TourFlow", "Этот файл появится после расчёта.", parent=self.root)
            return
        destination = filedialog.asksaveasfilename(
            parent=self.root, initialfile=source.name, defaultextension=source.suffix,
            filetypes=[(source.suffix[1:].upper(), f"*{source.suffix}"), ("Все файлы", "*.*")]
        )
        if destination:
            try:
                if Path(destination).resolve() != source.resolve():
                    shutil.copyfile(source, destination)
            except OSError as exc:
                messagebox.showerror("Не удалось сохранить файл", str(exc), parent=self.root)
            else:
                self.set_status(f"Файл сохранён · {Path(destination).name}")

    def open_results(self):
        if not self.results.is_dir():
            messagebox.showinfo("TourFlow", "Папка появится после расчёта.", parent=self.root)
            return
        try:
            if sys.platform == "win32":
                os.startfile(self.results)
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", self.results])
        except OSError as exc:
            messagebox.showerror("Не удалось открыть папку", str(exc), parent=self.root)

    def require_admin(self):
        if self.account is None:
            raise PermissionError("Войдите в приложение.")
        self.account.require_admin()

    def pick_excel(self):
        self.require_admin()
        filename = filedialog.askopenfilename(
            parent=self.root, title="Выберите выгрузку Росстата",
            filetypes=[("Файлы Excel", ("*.xlsx", "*.xlsm")), ("Все файлы", "*.*")]
        )
        if filename:
            self.excel.set(filename)

    def start_calculation(self, action):
        if self.account is None or self.running:
            return
        command = [sys.executable, "-u", str(tasks.ROOT / "gui_tasks.py"), action]
        if action == "update":
            try:
                self.require_admin()
                excel = Path(self.excel.get())
                tasks.check_excel(excel)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Обновление данных", str(exc), parent=self.root)
                return
            command.append(str(excel))
        self.log_lines = []
        self.running = True
        self.set_busy(True)
        self.set_status("Запуск расчёта…")
        if self.account.role == ADMIN:
            set_text(self.activity, "")
        threading.Thread(target=self.run_process, args=(action, command), daemon=True).start()

    def run_process(self, action, command):
        try:
            with subprocess.Popen(
                command, cwd=tasks.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            ) as child:
                for line in child.stdout:
                    self.events.put(("line", line))
                code = child.wait()
        except OSError as exc:
            self.events.put(("line", f"Не удалось запустить Python: {exc}\n"))
            code = 1
        self.events.put(("finished", (action, code)))

    def set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.analyze_button.configure(state=state)
        self.logout.configure(state=state)
        if self.account.role == ADMIN:
            self.update_button.configure(state=state)
            self.excel_button.configure(state=state)
        if busy:
            self.progress.pack(side="right")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def request_journal(self):
        self.require_admin()
        if self.reading_journal:
            return
        self.reading_journal = True
        self.journal_button.configure(state="disabled")
        self.journal_status.configure(text="Чтение журнала…", style="Hint.TLabel")

        def worker():
            try:
                self.events.put(("journal", (tasks.journal(), None)))
            except Exception as exc:
                self.events.put(("journal", (None, str(exc))))

        threading.Thread(target=worker, daemon=True).start()

    def fill_journal(self, rows, error):
        self.reading_journal = False
        self.journal_button.configure(state="normal")
        if error:
            self.journal_status.configure(text="Не удалось прочитать журнал PostgreSQL.",
                                          style="Error.TLabel")
            set_text(self.journal_detail, error)
            return
        self.journal_status.configure(
            text=f"Показано записей: {len(rows)}" if rows else "Журнал загрузок пока пуст.",
            style="Hint.TLabel"
        )
        self.journal_rows = {}
        self.journal_table.delete(*self.journal_table.get_children())
        for index, row in enumerate(rows):
            run_id, source, started, status, read, written, first, last, message = row
            period = " — ".join(filter(None, (first, last))) or "—"
            label = LOAD_STATUS.get(status, status)
            if status == "success" and source != "rosstat" and message:
                label = "С предупреждением"
            values = (tasks.SOURCES.get(source, source), local_time(started), label, read, written, period)
            item = self.journal_table.insert(
                "", "end", values=values, tags=("alternate",) if index % 2 else ()
            )
            self.journal_rows[item] = f"Запуск {run_id}\n{message or 'Дополнительных сообщений нет.'}"
        set_text(self.journal_detail, "")

    def show_journal_detail(self, _event=None):
        selection = self.journal_table.selection()
        if selection:
            set_text(self.journal_detail, self.journal_rows.get(selection[0], ""))

    def poll(self):
        # Ограничиваем число сообщений за один проход, чтобы окно реагировало на ввод.
        for _ in range(100):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                if value.startswith("@@PHASE "):
                    value = value.removeprefix("@@PHASE ")
                    self.set_status(value)
                self.log_lines.append(value)
                if self.account.role == ADMIN:
                    set_text(self.activity, value, append=True)
            elif kind == "finished":
                self.finish_calculation(*value)
            elif kind == "journal":
                self.fill_journal(*value)
        self.root.after(70, self.poll)

    def finish_calculation(self, action, code):
        self.running = False
        self.set_busy(False)
        if action == "update":
            self.request_journal()
        if code:
            self.set_status("Расчёт остановлен · доступны предыдущие результаты")
            self.show_calculation_error()
            return
        try:
            self.refresh_results()
            run = self.results / "run.json"
            metadata = json.loads(run.read_text(encoding="utf-8")) if run.is_file() else {}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.set_status(f"Не удалось открыть новые результаты. {exc}")
            return
        status = "Расчёт завершён"
        if metadata.get("warnings"):
            status += " · предупреждения при загрузке источников"
        self.set_status(status)
        if self.account.role == ADMIN:
            set_text(self.activity, f"\n{status}.\n", append=True)

    def show_calculation_error(self):
        window = tk.Toplevel(self.root)
        window.title("Расчёт остановлен")
        window.transient(self.root)
        window.geometry("780x420")
        window.grab_set()
        frame = ttk.Frame(window, padding=15)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Расчёт завершился с ошибкой.", style="Error.TLabel").pack(anchor="w")
        panel, log = text_panel(frame, 10)
        panel.pack(fill="both", expand=True, pady=8)
        error = next((line for line in reversed(self.log_lines)
                      if "Не удалось завершить операцию:" in line), "")
        if error:
            ttk.Label(frame, text=error.split("операцию:", 1)[-1].strip(), wraplength=720,
                      style="Hint.TLabel").pack(fill="x", before=panel, pady=(6, 0))
        set_text(log, "".join(self.log_lines))
        log.see("end")
        ttk.Button(frame, text="Закрыть", command=window.destroy).pack(anchor="e")

    def manage_accounts(self):
        self.require_admin()
        window = tk.Toplevel(self.root)
        window.title("Учётные записи")
        window.transient(self.root)
        window.grab_set()
        frame = ttk.Frame(window, padding=17)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Учётные записи", style="Page.TLabel").pack(anchor="w")
        panel, view = table(frame, ("Логин", "Роль"), 4, (185, 185))
        panel.pack(fill="x", pady=(7, 9))

        def refresh():
            set_rows(view, [(login, ROLES[role])
                            for login, role in self.accounts.list_users(self.account)])

        refresh()
        fields = ttk.Frame(frame)
        fields.pack(fill="x")
        entries = []
        for index, name in enumerate(("Логин", "Пароль", "Повторите пароль")):
            ttk.Label(fields, text=name).grid(row=index, column=0, sticky="w", pady=4)
            entry = ttk.Entry(fields, width=27, show="•" if index else "")
            entry.grid(row=index, column=1, padx=(9, 0), sticky="ew")
            entries.append(entry)
        ttk.Label(fields, text="Роль").grid(row=3, column=0, sticky="w", pady=4)
        role_names = (ROLES[ANALYST], ROLES[ADMIN])
        role_select = ttk.Combobox(fields, values=role_names, state="readonly", width=25)
        role_select.grid(row=3, column=1, padx=(9, 0), sticky="ew")
        role_select.current(0)
        notice = ttk.Label(frame, style="Hint.TLabel", wraplength=370)
        notice.pack(fill="x", pady=(8, 5))

        def add():
            login, password, repeated = (entry.get() for entry in entries)
            try:
                if password != repeated:
                    raise ValueError("Пароли не совпадают.")
                role = (ANALYST, ADMIN)[role_select.current()]
                created = self.accounts.create(login, password, role, self.account)
            except USER_ERRORS as exc:
                notice.configure(text=str(exc), style="Error.TLabel")
                return
            for entry in entries:
                entry.delete(0, "end")
            refresh()
            notice.configure(text=f"Создана учётная запись «{created.login}».", style="Success.TLabel")

        ttk.Button(frame, text="Создать", style="Primary.TButton", command=add).pack(anchor="e")
        entries[0].focus_set()

    def sign_out(self):
        if self.reading_journal:
            self.set_status("Дождитесь окончания чтения журнала.")
        elif not self.running:
            self.show_login()

    def close(self):
        if self.running:
            messagebox.showinfo("TourFlow", "Дождитесь окончания расчёта.", parent=self.root)
        else:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        TourFlow(root)
    except (OSError, sqlite3.Error) as exc:
        root.withdraw()
        messagebox.showerror("Не удалось запустить TourFlow", str(exc), parent=root)
        root.destroy()
        sys.exit(1)
    root.mainloop()
