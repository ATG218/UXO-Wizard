

import json
import os
import uuid
from datetime import datetime, timezone
from loguru import logger

class ProjectHistoryLogger:
    """
    Handles writing structured, append-only logs for a single project.
    """
    def __init__(self, project_root: str):
        self.log_dir = os.path.join(project_root, '_project', 'logs')
        self.log_file_path = os.path.join(self.log_dir, 'history.jsonl')
        self._file_handle = None
        self._ensure_log_dir_exists()
        self._open_log_file()

    def _ensure_log_dir_exists(self):
        """Create the log directory if it doesn't exist."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create log directory at {self.log_dir}: {e}")

    def _open_log_file(self):
        """Open the log file in append mode."""
        try:
            self._file_handle = open(self.log_file_path, 'a', encoding='utf-8')
        except IOError as e:
            logger.error(f"Failed to open project history file at {self.log_file_path}: {e}")
            self._file_handle = None

    def log_event(self, event_type: str, event_data: dict):
        """
        Writes a new event to the project's history log.

        Args:
            event_type (str): The type of event (e.g., 'processing.started').
            event_data (dict): A dictionary containing event-specific data.
        """
        if not self._file_handle:
            logger.warning("Cannot log project history event: log file is not open.")
            return

        log_entry = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "event_data": event_data
        }

        try:
            json_line = json.dumps(log_entry, ensure_ascii=False)
            self._file_handle.write(json_line + '\n')
            self._file_handle.flush()  # Ensure it's written immediately
        except (TypeError, IOError) as e:
            logger.error(f"Failed to write to project history log: {e}")

    def close(self):
        """Close the log file handle."""
        if self._file_handle:
            try:
                self._file_handle.close()
                self._file_handle = None
                logger.info("Project history logger closed.")
            except IOError as e:
                logger.error(f"Error closing project history file: {e}")


