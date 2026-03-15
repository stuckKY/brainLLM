"""Filesystem watcher that auto-triggers ingestion when documents change.

Uses watchdog to monitor DOCUMENTS_DIR. Changes are debounced — after the
last file event, the watcher waits DEBOUNCE_SECONDS before running ingestion.
This ensures batch file drops (e.g. uploading 10 files) trigger only one
ingestion run.

Thread-safe: a lock prevents concurrent ingestion runs.
"""

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from backend.ingest import DOCUMENTS_DIR, SUPPORTED_EXTENSIONS, run_ingestion

logger = logging.getLogger("brainllm.watcher")

DEBOUNCE_SECONDS = 5


class _IngestionHandler(FileSystemEventHandler):
    """Debounced handler that triggers ingestion on supported file changes."""

    def __init__(self) -> None:
        super().__init__()
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._ingesting = threading.Lock()

    def _is_relevant(self, path: str) -> bool:
        """Return True if the changed file has a supported extension."""
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS

    def _schedule_ingestion(self) -> None:
        """Reset the debounce timer. When it fires, run ingestion."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECONDS, self._run_ingestion)
            self._timer.daemon = True
            self._timer.start()

    def _run_ingestion(self) -> None:
        """Run the ingestion pipeline (guarded by a lock to prevent overlap)."""
        if not self._ingesting.acquire(blocking=False):
            logger.info("Auto-ingestion skipped — already running")
            return
        try:
            logger.info("Auto-ingestion triggered by file change")
            result = run_ingestion()
            new = result["files_new"]
            modified = result["files_modified"]
            deleted = result["files_deleted"]
            chunks = result["chunks_stored"]
            logger.info(
                "Auto-ingestion complete: new=%d modified=%d deleted=%d chunks=%d",
                new, modified, deleted, chunks,
            )
        except Exception:
            logger.exception("Auto-ingestion failed")
        finally:
            self._ingesting.release()

    # watchdog event callbacks
    def on_created(self, event):
        if not event.is_directory and self._is_relevant(event.src_path):
            logger.debug("File created: %s", event.src_path)
            self._schedule_ingestion()

    def on_modified(self, event):
        if not event.is_directory and self._is_relevant(event.src_path):
            logger.debug("File modified: %s", event.src_path)
            self._schedule_ingestion()

    def on_deleted(self, event):
        if not event.is_directory and self._is_relevant(event.src_path):
            logger.debug("File deleted: %s", event.src_path)
            self._schedule_ingestion()

    def on_moved(self, event):
        src_relevant = self._is_relevant(event.src_path)
        dest_relevant = self._is_relevant(event.dest_path)
        if not event.is_directory and (src_relevant or dest_relevant):
            logger.debug("File moved: %s → %s", event.src_path, event.dest_path)
            self._schedule_ingestion()


_observer: Observer | None = None


def start_watcher() -> None:
    """Start the filesystem watcher in a background thread.

    Safe to call multiple times — only starts once.
    """
    global _observer
    if _observer is not None:
        return

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    handler = _IngestionHandler()
    _observer = Observer()
    _observer.schedule(handler, str(DOCUMENTS_DIR), recursive=True)
    _observer.daemon = True
    _observer.start()

    logger.info("File watcher started — monitoring %s", DOCUMENTS_DIR)


def stop_watcher() -> None:
    """Stop the filesystem watcher."""
    global _observer
    if _observer is None:
        return

    _observer.stop()
    _observer.join(timeout=5)
    _observer = None
    logger.info("File watcher stopped")
