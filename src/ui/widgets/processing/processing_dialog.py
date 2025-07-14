"""
Processing Dialog for UXO Wizard - Modal dialog for data processing
"""
import os
import pickle
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QDialogButtonBox, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, Signal
import pandas as pd
from typing import Optional
from matplotlib.figure import Figure

from .processing_widget import ProcessingWidget
from ....processing import ProcessingResult


class ProcessingDialog(QDialog):
    """Modal dialog for data processing"""
    
    # Signals
    layer_created = Signal(object)  # UXOLayer created during processing
    plot_generated = Signal(Figure, str) # Figure, title

    def __init__(self, data: pd.DataFrame, parent=None, input_file_path: Optional[str] = None, project_manager=None):
        super().__init__(parent)
        self.data = data
        self.input_file_path = input_file_path
        self.project_manager = project_manager
        self.result: Optional[ProcessingResult] = None
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the dialog UI"""
        self.setWindowTitle("Data Processing")
        self.setModal(True)
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # Processing widget
        self.processing_widget = ProcessingWidget(self.project_manager)
        self.processing_widget.set_data(self.data, input_file_path=self.input_file_path)
        self.processing_widget.processing_complete.connect(self.on_processing_complete)
        
        # Forward layer creation signal
        self.processing_widget.layer_created.connect(self.layer_created.emit)
        
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

            # Handle generated plot if it exists
            if result.figure:
                self.handle_generated_plot(result.figure)

    def handle_generated_plot(self, figure: Figure):
        # Automatically show the plot in data viewer - no dialog needed
        # The pipeline already auto-saves plots to the processed directory
        self.plot_generated.emit(figure, "Processing Result Plot")

    def save_plot(self, figure: Figure):
        # Default to the 'processed/' folder if project is open
        start_dir = ""
        if self.project_manager:
            proj_dir = self.project_manager.get_current_working_directory()
            if proj_dir:
                start_dir = os.path.join(proj_dir, "processed")
                os.makedirs(start_dir, exist_ok=True)

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            start_dir,
            "Matplotlib Plots (*.mplplot)"
        )

        if filepath:
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(figure, f)
                QMessageBox.information(self, "Success", f"Plot saved to:{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot: {str(e)}")

    def get_result(self) -> Optional[ProcessingResult]:
        """Get the processing result"""
        return self.result 