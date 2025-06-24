"""
Console Widget for UXO Wizard - Log output and messages
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QToolBar, 
    QComboBox, QToolButton, QHBoxLayout, QLabel
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QFont
from loguru import logger
import sys
from datetime import datetime


class ConsoleWidget(QWidget):
    """Console widget for displaying logs and processing output"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_logger()
        
    def setup_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = QToolBar()
        
        # Log level filter
        self.level_combo = QComboBox()
        self.level_combo.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR"])
        self.level_combo.setCurrentText("INFO")
        self.level_combo.currentTextChanged.connect(self.filter_logs)
        toolbar.addWidget(QLabel("Level:"))
        toolbar.addWidget(self.level_combo)
        
        toolbar.addSeparator()
        
        # Clear button
        self.clear_btn = QToolButton()
        self.clear_btn.setText("Clear")
        self.clear_btn.clicked.connect(self.clear_console)
        toolbar.addWidget(self.clear_btn)
        
        # Auto-scroll toggle
        self.autoscroll_btn = QToolButton()
        self.autoscroll_btn.setText("Auto-scroll")
        self.autoscroll_btn.setCheckable(True)
        self.autoscroll_btn.setChecked(True)
        toolbar.addWidget(self.autoscroll_btn)
        
        # Text area
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 9))
        
        # Layout
        layout.addWidget(toolbar)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
        
        # Store formats for different log levels
        self.formats = {
            "DEBUG": self.create_format(QColor(150, 150, 150)),
            "INFO": self.create_format(QColor(200, 200, 200)),
            "WARNING": self.create_format(QColor(255, 200, 0)),
            "ERROR": self.create_format(QColor(255, 100, 100)),
            "CRITICAL": self.create_format(QColor(255, 0, 0), bold=True),
        }
        
    def create_format(self, color, bold=False):
        """Create text format for log level"""
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        if bold:
            fmt.setFontWeight(QFont.Bold)
        return fmt
        
    def setup_logger(self):
        """Configure loguru to output to this widget"""
        # Remove default handler
        logger.remove()
        
        # Add custom handler
        logger.add(
            self.write_log,
            format="{time:HH:mm:ss} | {level} | {message}",
            level="DEBUG"
        )
        
        # Also keep stderr for debugging
        logger.add(sys.stderr, level="ERROR")
        
    @Slot(str)
    def write_log(self, message):
        """Write log message to console"""
        # Parse log level from message
        level = "INFO"
        for lvl in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            if f"| {lvl} |" in message:
                level = lvl
                break
                
        # Check if we should display this level
        current_filter = self.level_combo.currentText()
        if current_filter != "ALL":
            level_order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if level_order.index(level) < level_order.index(current_filter):
                return
                
        # Move cursor to end
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Insert formatted text
        cursor.insertText(message + "\n", self.formats.get(level, self.formats["INFO"]))
        
        # Auto-scroll if enabled
        if self.autoscroll_btn.isChecked():
            self.text_edit.setTextCursor(cursor)
            self.text_edit.ensureCursorVisible()
            
    def clear_console(self):
        """Clear the console"""
        self.text_edit.clear()
        logger.info("Console cleared")
        
    def filter_logs(self, level):
        """Filter displayed logs by level"""
        # For now, just log the change
        # Full implementation would require storing all logs and re-filtering
        logger.info(f"Log filter changed to: {level}")
        
    def add_message(self, message, level="INFO"):
        """Add a message to the console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"{timestamp} | {level} | {message}"
        self.write_log(formatted_message) 