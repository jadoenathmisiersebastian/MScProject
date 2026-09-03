from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

from src.final_vision_pipeline import final_pipeline_config_fingerprint


STAGE_LINE = re.compile(
    r"^\[(\d+)/(\d+)\] "
    r"(Running|Completed|Skipping completed) stage: (.+)$"
)


@dataclass(frozen=True)
class DatasetReadiness:
    ready: bool
    message: str
    frame_count: int = 0


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def inspect_raw_dataset(path_value: str | Path) -> DatasetReadiness:
    if not str(path_value).strip():
        return DatasetReadiness(False, "Not selected")

    path = Path(path_value).expanduser()
    if not path.is_dir():
        return DatasetReadiness(False, "Directory not found")

    labels_path = path / "labels" / "vision_labels.jsonl"
    if not labels_path.is_file():
        return DatasetReadiness(False, "Missing labels/vision_labels.jsonl")

    try:
        with labels_path.open() as file:
            frame_count = sum(1 for line in file if line.strip())
    except OSError as error:
        return DatasetReadiness(False, f"Cannot read labels: {error}")

    if frame_count == 0:
        return DatasetReadiness(False, "Vision labels file is empty")

    return DatasetReadiness(True, f"Ready - {frame_count:,} frames", frame_count)


def load_editable_pipeline_config(
    config_path: str | Path,
    project_root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    root = Path(project_root).expanduser().resolve()
    path = resolve_project_path(root, config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline config does not exist: {path}")

    with path.open() as file:
        config = json.load(file)

    raw_splits = config.get("raw_splits", {})
    for split in ("train", "val", "test"):
        value = raw_splits.get(split, "")
        if value:
            raw_splits[split] = str(resolve_project_path(root, value))
    config["raw_splits"] = raw_splits
    return path, config


def save_pipeline_ui_config(
    config_path: str | Path,
    project_root: str | Path,
    pipeline_name: str,
    raw_splits: dict[str, str | Path],
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path, config = load_editable_pipeline_config(config_path, root)
    name = pipeline_name.strip()

    if not name:
        raise ValueError("Pipeline name is required.")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
        raise ValueError(
            "Pipeline name may contain only letters, numbers, underscores, and hyphens."
        )

    resolved_splits: dict[str, str] = {}
    for split in ("train", "val", "test"):
        value = str(raw_splits.get(split, "")).strip()
        if not value:
            raise ValueError(f"Select the {split} dataset directory.")
        resolved_splits[split] = str(resolve_project_path(root, value))

    config["pipeline_name"] = name
    config["raw_splits"] = resolved_splits
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(config, indent=2) + "\n")
    temporary_path.replace(path)

    fingerprint_config = dict(config)
    fingerprint_config["config_path"] = str(path)
    fingerprint_config["raw_splits"] = resolved_splits
    fingerprint_config["config_fingerprint"] = final_pipeline_config_fingerprint(
        fingerprint_config
    )
    return fingerprint_config


def saved_state_requires_restart(
    state_path: str | Path,
    fingerprint: str,
) -> bool:
    path = Path(state_path)
    if not path.is_file():
        return False

    try:
        with path.open() as file:
            state = json.load(file)
    except (OSError, json.JSONDecodeError):
        return True

    return state.get("config_fingerprint") != fingerprint


class FinalVisionPipelineWindow:
    def __init__(
        self,
        root: tk.Tk,
        project_root: str | Path,
        config_path: str | Path,
    ) -> None:
        self.root = root
        self.project_root = Path(project_root).expanduser().resolve()
        self.config_path, config = load_editable_pipeline_config(
            config_path,
            self.project_root,
        )
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_requested = False
        self.closing = False

        self.pipeline_name = tk.StringVar(
            value=str(config.get("pipeline_name", "c3d_final"))
        )
        raw_splits = config.get("raw_splits", {})
        self.split_paths = {
            split: tk.StringVar(value=str(raw_splits.get(split, "")))
            for split in ("train", "val", "test")
        }
        self.split_status = {
            split: tk.StringVar(value="Not checked")
            for split in ("train", "val", "test")
        }
        self.status_text = tk.StringVar(value="Ready")
        self.current_stage = tk.StringVar(value="No active stage")
        self.stage_count = tk.StringVar(value="0 / 0 stages")

        self._configure_window()
        self._configure_styles()
        self._build_layout()
        self._refresh_dataset_statuses()
        self._load_saved_progress()
        self.root.after(100, self._poll_output_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_window(self) -> None:
        self.root.title("Final RGB-D Vision Pipeline")
        self.root.geometry("1040x760")
        self.root.minsize(860, 650)
        self.root.configure(background="#f3f5f7")

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("aqua" if "aqua" in style.theme_names() else "clam")
        style.configure("App.TFrame", background="#f3f5f7")
        style.configure("Header.TFrame", background="#202a34")
        style.configure(
            "Header.TLabel",
            background="#202a34",
            foreground="#ffffff",
            font=("Helvetica Neue", 20, "bold"),
        )
        style.configure(
            "HeaderMeta.TLabel",
            background="#202a34",
            foreground="#c7d0d8",
            font=("Helvetica Neue", 11),
        )
        style.configure(
            "Section.TLabel",
            background="#f3f5f7",
            foreground="#202a34",
            font=("Helvetica Neue", 12, "bold"),
        )
        style.configure(
            "Field.TLabel",
            background="#f3f5f7",
            foreground="#34414d",
            font=("Helvetica Neue", 11),
        )
        style.configure(
            "Status.TLabel",
            background="#f3f5f7",
            foreground="#5b6772",
            font=("Helvetica Neue", 10),
        )
        style.configure(
            "CurrentStage.TLabel",
            background="#f3f5f7",
            foreground="#202a34",
            font=("Helvetica Neue", 12, "bold"),
        )
        style.configure("Pipeline.Horizontal.TProgressbar", thickness=12)

    def _build_layout(self) -> None:
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(24, 18))
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="Final RGB-D Vision Pipeline",
            style="Header.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text=str(self.config_path),
            style="HeaderMeta.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(
            header,
            textvariable=self.status_text,
            style="HeaderMeta.TLabel",
        ).grid(row=0, column=1, rowspan=2, sticky="e")

        inputs = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18, 24, 8))
        inputs.grid(row=1, column=0, sticky="ew")
        inputs.grid_columnconfigure(1, weight=1)
        ttk.Label(inputs, text="Dataset Configuration", style="Section.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 12)
        )

        ttk.Label(inputs, text="Pipeline name", style="Field.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 14), pady=5
        )
        ttk.Entry(inputs, textvariable=self.pipeline_name).grid(
            row=1, column=1, sticky="ew", pady=5
        )

        display_names = {
            "train": "Training data",
            "val": "Validation data",
            "test": "Test data",
        }
        for row, split in enumerate(("train", "val", "test"), start=2):
            ttk.Label(
                inputs,
                text=display_names[split],
                style="Field.TLabel",
            ).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=5)
            entry = ttk.Entry(inputs, textvariable=self.split_paths[split])
            entry.grid(row=row, column=1, sticky="ew", pady=5)
            entry.bind(
                "<FocusOut>",
                lambda _event, selected_split=split: self._refresh_dataset_status(
                    selected_split
                ),
            )
            ttk.Button(
                inputs,
                text="Browse...",
                command=lambda selected_split=split: self._choose_directory(
                    selected_split
                ),
            ).grid(row=row, column=2, padx=(10, 10), pady=5)
            ttk.Label(
                inputs,
                textvariable=self.split_status[split],
                style="Status.TLabel",
                width=27,
            ).grid(row=row, column=3, sticky="w", pady=5)

        progress_area = ttk.Frame(
            self.root,
            style="App.TFrame",
            padding=(24, 10, 24, 12),
        )
        progress_area.grid(row=2, column=0, sticky="ew")
        progress_area.grid_columnconfigure(0, weight=1)
        ttk.Label(
            progress_area,
            textvariable=self.current_stage,
            style="CurrentStage.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            progress_area,
            textvariable=self.stage_count,
            style="Status.TLabel",
        ).grid(row=0, column=1, sticky="e")
        self.progress = ttk.Progressbar(
            progress_area,
            mode="determinate",
            maximum=1,
            value=0,
            style="Pipeline.Horizontal.TProgressbar",
        )
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 12))

        actions = ttk.Frame(progress_area, style="App.TFrame")
        actions.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.start_button = ttk.Button(
            actions,
            text="Start / Resume",
            command=lambda: self._prepare_run(restart=False),
        )
        self.start_button.pack(side="left")
        self.restart_button = ttk.Button(
            actions,
            text="Restart All",
            command=lambda: self._prepare_run(restart=True),
        )
        self.restart_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(
            actions,
            text="Stop",
            command=self._request_stop,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(
            actions,
            text="Open Results",
            command=self._open_results,
        ).pack(side="right")

        log_area = ttk.Frame(self.root, style="App.TFrame", padding=(24, 0, 24, 20))
        log_area.grid(row=3, column=0, sticky="nsew")
        log_area.grid_rowconfigure(1, weight=1)
        log_area.grid_columnconfigure(0, weight=1)
        ttk.Label(log_area, text="Live Output", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.log = scrolledtext.ScrolledText(
            log_area,
            wrap="word",
            state="disabled",
            background="#111820",
            foreground="#d8e0e7",
            insertbackground="#ffffff",
            selectbackground="#31576d",
            borderwidth=0,
            padx=12,
            pady=10,
            font=("Menlo", 10),
        )
        self.log.grid(row=1, column=0, sticky="nsew")
        self.log.tag_configure("stage", foreground="#74c7ec")
        self.log.tag_configure("error", foreground="#ff8f8f")
        self.log.tag_configure("success", foreground="#89d185")

    def _choose_directory(self, split: str) -> None:
        current = self.split_paths[split].get().strip()
        initial_directory = current if Path(current).is_dir() else self.project_root
        selected = filedialog.askdirectory(
            parent=self.root,
            title=f"Select {split} Unity dataset",
            initialdir=initial_directory,
            mustexist=True,
        )
        if selected:
            self.split_paths[split].set(selected)
            self._refresh_dataset_status(split)

    def _refresh_dataset_status(self, split: str) -> DatasetReadiness:
        readiness = inspect_raw_dataset(self.split_paths[split].get())
        self.split_status[split].set(readiness.message)
        return readiness

    def _refresh_dataset_statuses(self) -> None:
        for split in ("train", "val", "test"):
            self._refresh_dataset_status(split)

    def _state_path(self) -> Path:
        name = self.pipeline_name.get().strip() or "c3d_final"
        return self.project_root / "reports" / "pipelines" / name / "pipeline_state.json"

    def _load_saved_progress(self) -> None:
        state_path = self._state_path()
        if not state_path.is_file():
            self.progress.configure(maximum=1, value=0)
            self.stage_count.set("0 / 0 stages")
            return

        try:
            with state_path.open() as file:
                state = json.load(file)
        except (OSError, json.JSONDecodeError):
            return

        total = int(state.get("total_stages", 0))
        completed = len(state.get("completed_stages", []))
        self.progress.configure(maximum=max(total, 1), value=completed)
        self.stage_count.set(f"{completed} / {total} stages")
        status = state.get("status", "pending")
        if status == "complete":
            self.status_text.set("Complete")
            self.current_stage.set("Pipeline complete")
        elif status == "failed":
            failed_stage = state.get("failed_stage") or "unknown stage"
            self.status_text.set("Failed")
            self.current_stage.set(f"Failed: {self._display_stage(failed_stage)}")
        elif state.get("current_stage"):
            self.status_text.set("Interrupted")
            self.current_stage.set(
                f"Last stage: {self._display_stage(state['current_stage'])}"
            )
        elif completed:
            self.status_text.set("Ready to resume")
            self.current_stage.set("Saved progress available")

    def _prepare_run(self, restart: bool) -> None:
        if self.process is not None:
            return

        readiness = {
            split: self._refresh_dataset_status(split)
            for split in ("train", "val", "test")
        }
        invalid = [split for split, result in readiness.items() if not result.ready]
        if invalid:
            messagebox.showerror(
                "Dataset Not Ready",
                "Check the selected train, validation, and test directories before running.",
                parent=self.root,
            )
            return

        try:
            config = save_pipeline_ui_config(
                config_path=self.config_path,
                project_root=self.project_root,
                pipeline_name=self.pipeline_name.get(),
                raw_splits={
                    split: variable.get()
                    for split, variable in self.split_paths.items()
                },
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            messagebox.showerror("Invalid Configuration", str(error), parent=self.root)
            return

        state_path = self._state_path()
        restart_already_confirmed = False
        if not restart and saved_state_requires_restart(
            state_path,
            str(config["config_fingerprint"]),
        ):
            restart = messagebox.askyesno(
                "Dataset Configuration Changed",
                "Saved progress belongs to different datasets or settings. "
                "Restart this pipeline from stage 1?",
                parent=self.root,
            )
            if not restart:
                return
            restart_already_confirmed = True

        if restart and state_path.exists() and not restart_already_confirmed:
            confirmed = messagebox.askyesno(
                "Restart Entire Pipeline",
                "Rerun every stage and replace outputs for this pipeline name?",
                parent=self.root,
            )
            if not confirmed:
                return

        self._start_process(restart)

    def _start_process(self, restart: bool) -> None:
        command = [
            sys.executable,
            "-u",
            str(self.project_root / "main.py"),
            "--run-final-vision-pipeline",
            "--pipeline-config",
            str(self.config_path),
        ]
        if restart:
            command.append("--pipeline-restart")

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process_options: dict[str, Any] = {}
        if os.name == "posix":
            process_options["start_new_session"] = True

        self.stop_requested = False
        self._set_running_controls(True)
        self.status_text.set("Running")
        self.current_stage.set("Starting pipeline")
        self._append_log("\n$ " + " ".join(command) + "\n", "stage")

        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
                **process_options,
            )
        except OSError as error:
            self.process = None
            self._set_running_controls(False)
            self.status_text.set("Failed to start")
            messagebox.showerror("Could Not Start Pipeline", str(error), parent=self.root)
            return

        threading.Thread(target=self._read_process_output, daemon=True).start()

    def _read_process_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return

        for line in process.stdout:
            self.output_queue.put(("line", line.rstrip("\n")))
        return_code = process.wait()
        self.output_queue.put(("exit", return_code))

    def _poll_output_queue(self) -> None:
        try:
            while True:
                event, value = self.output_queue.get_nowait()
                if event == "line":
                    self._handle_output_line(str(value))
                elif event == "exit":
                    self._handle_process_exit(int(value))
        except queue.Empty:
            pass

        if not self.closing or self.process is not None:
            self.root.after(100, self._poll_output_queue)

    def _handle_output_line(self, line: str) -> None:
        match = STAGE_LINE.match(line)
        tag = "error" if "error" in line.lower() or "traceback" in line.lower() else None
        if match:
            index = int(match.group(1))
            total = int(match.group(2))
            action = match.group(3)
            stage = match.group(4)
            completed = index if action != "Running" else index - 1
            self.progress.configure(maximum=max(total, 1), value=max(completed, 0))
            self.stage_count.set(f"{max(completed, 0)} / {total} stages")
            if action == "Running":
                self.current_stage.set(self._display_stage(stage))
            tag = "stage"
        elif line.startswith("Final vision pipeline complete"):
            tag = "success"

        self._append_log(line + "\n", tag)

    def _handle_process_exit(self, return_code: int) -> None:
        self.process = None
        self._set_running_controls(False)
        if self.stop_requested:
            self.status_text.set("Stopped")
            self.current_stage.set("Pipeline stopped - progress was saved")
            self._append_log("Pipeline stopped. Run Start / Resume to continue.\n", "error")
        elif return_code == 0:
            maximum = int(float(self.progress.cget("maximum")))
            self.progress.configure(value=maximum)
            self.stage_count.set(f"{maximum} / {maximum} stages")
            self.status_text.set("Complete")
            self.current_stage.set("Pipeline complete")
            self._append_log("Pipeline completed successfully.\n", "success")
        else:
            self.status_text.set("Failed")
            self.current_stage.set("Pipeline failed - inspect the live output")
            self._append_log(
                f"Pipeline exited with code {return_code}.\n",
                "error",
            )

        if self.closing:
            self.root.destroy()

    def _request_stop(self) -> None:
        if self.process is None:
            return
        if not messagebox.askyesno(
            "Stop Pipeline",
            "Stop the current stage? Completed stages will remain resumable.",
            parent=self.root,
        ):
            return

        self.stop_requested = True
        self.status_text.set("Stopping")
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
        except (OSError, ProcessLookupError):
            pass
        self.root.after(5000, self._force_stop_if_needed)

    def _force_stop_if_needed(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            else:
                self.process.kill()
        except (OSError, ProcessLookupError):
            pass

    def _set_running_controls(self, running: bool) -> None:
        normal_or_disabled = "disabled" if running else "normal"
        self.start_button.configure(state=normal_or_disabled)
        self.restart_button.configure(state=normal_or_disabled)
        self.stop_button.configure(state="normal" if running else "disabled")

    def _append_log(self, text: str, tag: str | None = None) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text, tag or ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def _open_results(self) -> None:
        results = self.project_root / "reports" / "pipelines" / self.pipeline_name.get().strip()
        if not results.exists():
            messagebox.showinfo(
                "No Results Yet",
                "The results directory will be created when the pipeline starts.",
                parent=self.root,
            )
            return

        if sys.platform == "darwin":
            subprocess.Popen(["open", str(results)])
        elif os.name == "nt":
            os.startfile(results)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(results)])

    def _on_close(self) -> None:
        if self.process is None:
            self.root.destroy()
            return

        if not messagebox.askyesno(
            "Close Pipeline Window",
            "Stop the running pipeline and close the window?",
            parent=self.root,
        ):
            return

        self.closing = True
        self.stop_requested = True
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
        except (OSError, ProcessLookupError):
            self.root.destroy()
            return

        self.root.after(5000, self._force_stop_if_needed)

    @staticmethod
    def _display_stage(stage: str) -> str:
        return stage.replace("_", " ").title()


def launch_final_pipeline_ui(
    config_path: str | Path = "config/final_vision_pipeline.json",
    project_root: str | Path | None = None,
) -> None:
    root_path = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    window = tk.Tk()
    FinalVisionPipelineWindow(window, root_path, config_path)
    window.mainloop()
