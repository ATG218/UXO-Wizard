"""
Project Relocation Dialog for handling projects with inaccessible working directories
"""

import os
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QTextEdit, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class ProjectRelocationDialog(QDialog):
    """Dialog to help users relocate projects to accessible directories"""
    
    def __init__(self, original_path: str, project_name: str, parent=None):
        super().__init__(parent)
        self.original_path = original_path
        self.project_name = project_name
        self.selected_path = None
        
        self.setWindowTitle("Project Relocation Required")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Project Relocation Required")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Explanation
        explanation = QLabel(
            "This project was created on a different machine and its original "
            "working directory is not accessible on this system."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        
        # Original path group
        original_group = QGroupBox("Original Project Location")
        original_layout = QVBoxLayout(original_group)
        
        original_label = QLabel("The project was originally located at:")
        original_layout.addWidget(original_label)
        
        # Original path display
        self.original_path_display = QTextEdit()
        self.original_path_display.setPlainText(self.original_path)
        self.original_path_display.setMaximumHeight(60)
        self.original_path_display.setReadOnly(True)
        original_layout.addWidget(self.original_path_display)
        
        layout.addWidget(original_group)
        
        # New location group
        new_group = QGroupBox("Choose New Project Location")
        new_layout = QVBoxLayout(new_group)
        
        new_label = QLabel(
            "Please choose where you want to work with this project on your machine.\n"
            "The project files and history will be restored to this location."
        )
        new_label.setWordWrap(True)
        new_layout.addWidget(new_label)
        
        # Selected path display
        self.selected_path_display = QTextEdit()
        self.selected_path_display.setPlainText("No location selected")
        self.selected_path_display.setMaximumHeight(60)
        self.selected_path_display.setReadOnly(True)
        new_layout.addWidget(self.selected_path_display)
        
        # Browse buttons
        button_layout = QHBoxLayout()
        
        self.browse_button = QPushButton("Browse for Location...")
        self.browse_button.clicked.connect(self.browse_for_location)
        button_layout.addWidget(self.browse_button)
        
        self.current_dir_button = QPushButton("Use Current Directory")
        self.current_dir_button.clicked.connect(self.use_current_directory)
        button_layout.addWidget(self.current_dir_button)
        
        new_layout.addLayout(button_layout)
        layout.addWidget(new_group)
        
        # Warning
        warning = QLabel(
            "⚠️ Note: This will create a '_project' folder structure in the chosen location "
            "to store project logs and metadata."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #D68000; font-weight: bold;")
        layout.addWidget(warning)
        
        # Dialog buttons
        dialog_buttons = QHBoxLayout()
        
        self.ok_button = QPushButton("Continue")
        self.ok_button.clicked.connect(self.accept_relocation)
        self.ok_button.setEnabled(False)  # Disabled until location selected
        dialog_buttons.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        dialog_buttons.addWidget(self.cancel_button)
        
        dialog_buttons.addStretch()
        layout.addLayout(dialog_buttons)
        
    def browse_for_location(self):
        """Open file dialog to browse for new location"""
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose Project Location",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if selected_dir:
            # Create project subdirectory
            project_dir = Path(selected_dir) / self.project_name
            self.selected_path = str(project_dir)
            self.selected_path_display.setPlainText(str(project_dir))
            self.ok_button.setEnabled(True)
            
    def use_current_directory(self):
        """Use current working directory"""
        current_dir = Path.cwd() / self.project_name
        self.selected_path = str(current_dir)
        self.selected_path_display.setPlainText(str(current_dir))
        self.ok_button.setEnabled(True)
        
    def accept_relocation(self):
        """Accept the selected location after validation"""
        if not self.selected_path:
            QMessageBox.warning(self, "No Location", "Please select a location for the project.")
            return
            
        # Validate the selected path
        selected_path = Path(self.selected_path)
        
        try:
            # Test if we can create the directory
            selected_path.mkdir(parents=True, exist_ok=True)
            
            # Test if we can write to it
            test_file = selected_path / ".test_write"
            test_file.touch()
            test_file.unlink()
            
            self.accept()
            
        except (PermissionError, OSError) as e:
            QMessageBox.critical(
                self, 
                "Access Error", 
                f"Cannot create project at selected location:\n\n{e}\n\nPlease choose a different location."
            )
            
    def get_selected_path(self) -> str:
        """Get the selected path"""
        return self.selected_path