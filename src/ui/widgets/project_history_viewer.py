
import json
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QTextEdit, QSplitter, QListWidgetItem, 
    QHBoxLayout, QPushButton
)
from PySide6.QtCore import Qt
from loguru import logger

from ...project.project_manager import ProjectManager

class ProjectHistoryViewer(QWidget):
    """A widget to display the project's history log."""

    def __init__(self, project_manager: ProjectManager, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.setup_ui()

    def setup_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Project History")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Add refresh button
        button_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh History")
        self.refresh_button.clicked.connect(self.load_history)
        button_layout.addWidget(self.refresh_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        splitter = QSplitter(Qt.Horizontal)

        self.event_list = QListWidget()
        self.event_list.currentItemChanged.connect(self.display_event_details)

        self.details_view = QTextEdit()
        self.details_view.setReadOnly(True)
        self.details_view.setLineWrapMode(QTextEdit.NoWrap)

        splitter.addWidget(self.event_list)
        splitter.addWidget(self.details_view)
        splitter.setSizes([200, 600])

        layout.addWidget(splitter)
        self.setLayout(layout)

    def load_history(self):
        """Load and display the history from the project's log file."""
        logger.info("ProjectHistoryViewer: load_history() called")
        self.clear_history()
        project_root = self.project_manager.get_current_working_directory()
        logger.info(f"ProjectHistoryViewer: project_root = {project_root}")
        if not project_root:
            logger.warning("Cannot load project history: no project is open.")
            return

        log_file = os.path.join(project_root, '_project', 'logs', 'history.jsonl')
        logger.info(f"ProjectHistoryViewer: Looking for log file at {log_file}")

        if not os.path.exists(log_file):
            logger.warning(f"No project history file found at {log_file}")
            return

        event_count = 0
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        summary = self._format_event_summary(event)
                        logger.info(f"ProjectHistoryViewer: Adding event: {summary}")
                        item = QListWidgetItem(summary)
                        item.setData(Qt.UserRole, event)  # Store full event data
                        self.event_list.insertItem(0, item) # Insert at top for chronological order
                        event_count += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed line in history log: {line[:100]}... Error: {e}")
            logger.info(f"ProjectHistoryViewer: Successfully loaded {event_count} events")
        except IOError as e:
            logger.error(f"Failed to read project history file: {e}")

    def clear_history(self):
        """Clear the history view."""
        self.event_list.clear()
        self.details_view.clear()

    def display_event_details(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        """Display the full JSON data for the selected event."""
        if not current_item:
            return

        event_data = current_item.data(Qt.UserRole)
        pretty_json = json.dumps(event_data, indent=2)
        self.details_view.setText(pretty_json)

    def _format_event_summary(self, event: dict) -> str:
        """Create a human-readable summary for an event."""
        event_type = event.get('event_type', 'unknown')
        event_data = event.get('event_data', {})
        
        # Safe timestamp parsing
        timestamp_str = event.get('timestamp', '')
        try:
            if 'T' in timestamp_str and '.' in timestamp_str:
                timestamp = timestamp_str.split('T')[1].split('.')[0]
            elif 'T' in timestamp_str:
                timestamp = timestamp_str.split('T')[1][:8]  # Take first 8 chars (HH:MM:SS)
            else:
                timestamp = timestamp_str[:8] if len(timestamp_str) >= 8 else timestamp_str
        except (IndexError, AttributeError):
            timestamp = "??:??:??"

        summary = f"[{timestamp}] "

        if event_type == 'project.created':
            summary += f"✅ Project Created: {event_data.get('project_name', '')}"
        elif event_type == 'project.loaded':
            summary += "📂 Project Loaded"
        elif event_type == 'processing.started':
            summary += f"⚙️ Started: {event_data.get('script_id', 'Unknown Script')}"
        elif event_type == 'processing.completed':
            status_icon = "✔️" if event_data.get('status') == 'success' else "❌"
            # Try multiple possible field names for script identification
            metadata = event_data.get('metadata', {})
            
            # Check new enhanced metadata structure first
            execution_details = metadata.get('execution_details', {})
            script_metadata = metadata.get('script_metadata', {})
            parameters = metadata.get('parameters', {})
            script_selection = parameters.get('script_selection', {})
            script_name = script_selection.get('script_name', {})
            
            script_id = (execution_details.get('processing_script') or  # New enhanced metadata
                        script_metadata.get('script') or              # New enhanced metadata
                        metadata.get('script') or                     # Legacy direct metadata
                        metadata.get('processing_script') or          # Legacy direct metadata
                        script_name.get('value') or                   # Parameter-based lookup
                        event_data.get('script_id') or                # Fallback to event data
                        'Unknown Script')
            summary += f"{status_icon} Finished: {script_id}"
        else:
            summary += event_type

        return summary
