"""
Processing Dialog for UXO Wizard - Modal dialog for data processing
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QDialogButtonBox
)
from PySide6.QtCore import Qt
import pandas as pd
from typing import Optional

from .processing_widget import ProcessingWidget
from ..processing import ProcessingResult


class ProcessingDialog(QDialog):
    """Modal dialog for data processing"""
    
    def __init__(self, data: pd.DataFrame, parent=None, input_file_path: Optional[str] = None):
        super().__init__(parent)
        self.data = data
        self.input_file_path = input_file_path
        self.result: Optional[ProcessingResult] = None
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the dialog UI"""
        self.setWindowTitle("Data Processing")
        self.setModal(True)
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # Processing widget
        self.processing_widget = ProcessingWidget()
        self.processing_widget.set_data(self.data, input_file_path=self.input_file_path)
        self.processing_widget.processing_complete.connect(self.on_processing_complete)
        layout.addWidget(self.processing_widget)
        
        # Dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Close
        )
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
        self.setLayout(layout)
        
    def on_processing_complete(self, result: ProcessingResult):
        """Handle processing completion"""
        self.result = result
        if result.success:
            # Change Close button to OK
            self.button_box.clear()
            self.button_box.setStandardButtons(
                QDialogButtonBox.Ok | QDialogButtonBox.Close
            )
            self.button_box.accepted.connect(self.accept)
            self.button_box.rejected.connect(self.reject)
            
    def get_result(self) -> Optional[ProcessingResult]:
        """Get the processing result"""
        return self.result 