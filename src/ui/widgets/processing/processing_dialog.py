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
    processing_complete = Signal(ProcessingResult)  # Emitted when processing completes

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
        
        # Emit signal for external listeners (e.g., history viewer refresh)
        self.processing_complete.emit(result)
        
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
    
    def configure_from_previous(self, processor_type: str, script_name: str, parameters: dict):
        """Configure dialog to use specific processor, script, and parameters"""
        try:
            # Find the processor by type
            processors = self.processing_widget.pipeline.get_available_processors()
            processor_id = None
            for proc_info in processors:
                if proc_info['id'] == processor_type:
                    processor_id = proc_info['id']
                    break
            
            if not processor_id:
                raise ValueError(f"Processor type '{processor_type}' not found")
            
            # Select the processor (this will populate parameters)
            self.processing_widget.select_processor(processor_id)
            
            # Wait for UI to be ready and get the current processor
            from PySide6.QtCore import QCoreApplication
            QCoreApplication.processEvents()
            
            processor = self.processing_widget.current_processor
            if not processor:
                raise ValueError("Failed to get current processor")
            
            # Set the script name first
            processor.set_script(script_name)
            
            # Now apply all the saved parameters to the processor
            self._apply_parameters_to_processor(processor, parameters)
            
            # Recreate the parameter widget with the updated parameters
            if hasattr(self.processing_widget, 'params_widget') and self.processing_widget.params_widget:
                # Store flag to prevent recursive updates during parameter restoration
                self.processing_widget._updating_parameters = True
                try:
                    # Remove old widget
                    old_widget = self.processing_widget.params_scroll.widget()
                    if old_widget:
                        self.processing_widget.params_scroll.takeWidget()
                        old_widget.setParent(None)
                        old_widget.deleteLater()
                    
                    # Process events to ensure cleanup
                    QCoreApplication.processEvents()
                    
                    # Create new parameter widget with updated parameters
                    from .processing_widget import ParameterWidget
                    new_params_widget = ParameterWidget(processor.parameters)
                    new_params_widget.value_changed.connect(self.processing_widget.on_parameter_changed)
                    
                    # Set the new widget
                    self.processing_widget.params_scroll.setWidget(new_params_widget)
                    self.processing_widget.params_widget = new_params_widget
                    
                    # Enhance script dropdown
                    self.processing_widget._enhance_script_dropdown(processor)
                    
                finally:
                    self.processing_widget._updating_parameters = False
            
            # Force the processing widget to show the processing view
            self.processing_widget.show_processing_view()
            
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", f"Failed to configure processing: {str(e)}")
            return False
    
    def _apply_parameters_to_processor(self, processor, saved_parameters: dict):
        """Apply saved parameters to the processor's parameter structure"""
        try:
            # Apply saved parameters to processor's parameters structure
            for category, category_params in saved_parameters.items():
                if category in processor.parameters:
                    for param_name, param_data in category_params.items():
                        if param_name in processor.parameters[category] and 'value' in param_data:
                            processor.parameters[category][param_name]['value'] = param_data['value']
        except Exception as e:
            # Log but don't fail - use default parameters if restoration fails
            from loguru import logger
            logger.warning(f"Failed to restore some parameters: {e}")
            pass
    
    def start_processing_silently(self):
        """Start processing without user interaction"""
        try:
            # Trigger the start processing method on the processing widget
            self.processing_widget.start_processing()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"Failed to start processing: {str(e)}")
            return False