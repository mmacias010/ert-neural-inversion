"""Small command-line progress helpers."""

from __future__ import annotations

import sys
import time
from typing import Any, TextIO


class InversionProgressPrinter:
    """Render inversion progress callback events to a terminal or log stream."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        stream: TextIO | None = None,
        width: int = 28,
        min_interval_sec: float = 0.1,
    ) -> None:
        self.enabled = bool(enabled)
        self.stream = stream if stream is not None else sys.stderr
        self.width = int(width)
        self.min_interval_sec = float(min_interval_sec)
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._status_len = 0
        self._last_status_at = 0.0
        self._window_index: int | None = None
        self._n_windows: int | None = None

    def __call__(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        event = str(payload.get("event", ""))
        if event == "single_start":
            self._line(
                "Single inversion: "
                f"{int(payload['n_cells'])} cells, "
                f"{int(payload['n_data'])} data, "
                f"max_iterations={int(payload['max_iterations'])}"
            )
        elif event == "single_iteration_start":
            iteration = int(payload["iteration"])
            max_iterations = int(payload["max_iterations"])
            self._status(
                f"Single inversion {self._bar(iteration - 1, max_iterations)} "
                f"iter {iteration}/{max_iterations}: solving"
            )
        elif event == "single_iteration_done":
            self._status(
                self._iteration_message("Single inversion", payload),
                force=True,
                log_when_not_tty=True,
            )
        elif event == "single_done":
            self._line(
                "Single inversion done: "
                f"iterations={int(payload['iterations'])}/{int(payload['max_iterations'])}, "
                f"final chi2={self._value(payload.get('final_chi2'))}, "
                f"stop={payload.get('stop_reason', 'unknown')}"
            )
        elif event == "timelapse_start":
            if self._window_index is None:
                self._line(
                    "Full time-lapse inversion: "
                    f"{int(payload['n_times'])} timesteps, "
                    f"{int(payload['n_cells'])} cells, "
                    f"max_iterations={int(payload['max_iterations'])}"
                )
        elif event == "timelapse_iteration_start":
            self._status(self._iteration_start_message(payload))
        elif event == "timelapse_time_start":
            self._status(
                self._time_step_message(payload),
                log_when_not_tty=self._window_index is None and self._is_time_milestone(payload),
            )
        elif event == "timelapse_iteration_done":
            self._status(
                self._iteration_message(self._timelapse_prefix(payload), payload),
                force=True,
                log_when_not_tty=self._window_index is None,
            )
        elif event == "timelapse_done":
            if self._window_index is None:
                self._line(
                    "Full time-lapse inversion done: "
                    f"iterations={int(payload['iterations'])}/{int(payload['max_iterations'])}, "
                    f"final chi2={self._value(payload.get('final_chi2'))}, "
                    f"stop={payload.get('stop_reason', 'unknown')}"
                )
        elif event == "windowed_start":
            self._n_windows = int(payload["n_windows"])
            self._line(
                "Windowed inversion: "
                f"{self._n_windows} windows, "
                f"window_size={int(payload['window_size'])}, "
                f"window_step={int(payload['window_step'])}, "
                f"max_iterations={int(payload['max_iterations'])}"
            )
        elif event == "window_start":
            self._window_index = int(payload["window_index"])
            self._n_windows = int(payload["n_windows"])
            self._status(
                f"Windowed {self._bar(self._window_index - 1, self._n_windows)} "
                f"window {self._window_index}/{self._n_windows} "
                f"steps {int(payload['start_idx'])}-{int(payload['end_idx'])}"
            )
        elif event == "window_done":
            window_index = int(payload["window_index"])
            n_windows = int(payload["n_windows"])
            self._status(
                f"Windowed {self._bar(window_index, n_windows)} "
                f"window {window_index}/{n_windows} done: "
                f"steps {int(payload['start_idx'])}-{int(payload['end_idx'])}, "
                f"chi2={self._value(payload.get('final_chi2'))}",
                force=True,
                log_when_not_tty=True,
            )
            self._window_index = None
        elif event == "windowed_prediction_start":
            self._status("Windowed inversion: building final predicted responses", force=True)
        elif event == "windowed_prediction_step":
            time_number = int(payload["time_number"])
            n_times = int(payload["n_times"])
            self._status(
                f"Windowed prediction {self._bar(time_number, n_times)} "
                f"timestep {time_number}/{n_times}"
            )
        elif event == "windowed_done":
            self._line(
                "Windowed inversion done: "
                f"windows={int(payload['n_windows'])}, "
                f"final window chi2={self._value(payload.get('final_chi2'))}"
            )

    def finish(self) -> None:
        if self.enabled and self._is_tty and self._status_len:
            self.stream.write("\n")
            self.stream.flush()
            self._status_len = 0

    def _iteration_start_message(self, payload: dict[str, Any]) -> str:
        iteration = int(payload["iteration"])
        max_iterations = int(payload["max_iterations"])
        return (
            f"{self._timelapse_prefix(payload)} {self._bar(iteration - 1, max_iterations)} "
            f"iter {iteration}/{max_iterations}: solving"
        )

    def _iteration_message(self, prefix: str, payload: dict[str, Any]) -> str:
        iteration = int(payload["iteration"])
        max_iterations = int(payload["max_iterations"])
        return (
            f"{prefix} {self._bar(iteration, max_iterations)} "
            f"iter {iteration}/{max_iterations}: "
            f"chi2={self._value(payload.get('chi2'))}, "
            f"step={self._value(payload.get('step_norm'))}"
        )

    def _time_step_message(self, payload: dict[str, Any]) -> str:
        stage = {
            "linearization": "linearization",
            "candidate": "candidate",
            "line_search": "line search",
        }.get(str(payload.get("stage", "")), str(payload.get("stage", "step")))
        time_number = int(payload["time_number"])
        n_times = int(payload["n_times"])
        iteration = int(payload["iteration"])
        max_iterations = int(payload["max_iterations"])
        return (
            f"{self._timelapse_prefix(payload)} {self._bar(iteration - 1, max_iterations)} "
            f"iter {iteration}/{max_iterations}: "
            f"{stage} {time_number}/{n_times}"
        )

    def _is_time_milestone(self, payload: dict[str, Any]) -> bool:
        time_number = int(payload["time_number"])
        n_times = max(1, int(payload["n_times"]))
        stride = max(1, n_times // 10)
        return time_number == 1 or time_number == n_times or time_number % stride == 0

    def _timelapse_prefix(self, payload: dict[str, Any]) -> str:
        if self._window_index is None or self._n_windows is None:
            return "Full time-lapse"
        return f"Windowed window {self._window_index}/{self._n_windows}"

    def _bar(self, done: int, total: int) -> str:
        total = max(1, int(total))
        done = min(max(0, int(done)), total)
        filled = int(round(self.width * done / total))
        return "[" + "#" * filled + "." * (self.width - filled) + "]"

    def _value(self, value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value)

    def _line(self, message: str) -> None:
        if not self.enabled:
            return
        self._clear_status()
        print(message, file=self.stream, flush=True)

    def _status(self, message: str, *, force: bool = False, log_when_not_tty: bool = False) -> None:
        if not self.enabled:
            return
        if not self._is_tty:
            if log_when_not_tty:
                self._line(message)
            return
        now = time.monotonic()
        if not force and now - self._last_status_at < self.min_interval_sec:
            return
        self._last_status_at = now
        padding = " " * max(0, self._status_len - len(message))
        self.stream.write("\r" + message + padding)
        self.stream.flush()
        self._status_len = len(message)

    def _clear_status(self) -> None:
        if self._is_tty and self._status_len:
            self.stream.write("\r" + " " * self._status_len + "\r")
            self.stream.flush()
            self._status_len = 0
