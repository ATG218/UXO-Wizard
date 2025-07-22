"""
Processing Widget for UXO Wizard - Interactive data processing with animations
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QProgressBar, QGroupBox, QScrollArea,
    QCheckBox, QSlider, QSpinBox, QDoubleSpinBox, QFrame,
    QGraphicsOpacityEffect, QStackedWidget, QLineEdit, QFileDialog
)
from PySide6.QtCore import (
    Qt, Signal, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup, QSequentialAnimationGroup,
    QTimer, Property, QCoreApplication
)
from PySide6.QtGui import QPalette, QFont
import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger

from ....processing import ProcessingPipeline, ProcessingResult


class AnimatedProgressBar(QProgressBar):
    """Custom progress bar with smooth animations"""
    
    def __init__(self):
        super().__init__()
        self.setTextVisible(True)
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3daee9;
                border-radius: 5px;
                text-align: center;
                background-color: #232629;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #3daee9, stop: 1 #1d99f3);
                border-radius: 3px;
            }
        """)
        
        # Pulse animation for indeterminate state
        self._pulse_animation = QPropertyAnimation(self, b"value")
        self._pulse_animation.setDuration(1000)
        self._pulse_animation.setLoopCount(-1)
        
    def set_indeterminate(self, indeterminate: bool = True):
        """Set progress bar to indeterminate mode with animation"""
        if indeterminate:
            self.setRange(0, 0)
        else:
            self.setRange(0, 100)
            self._pulse_animation.stop()


class ProcessorCard(QFrame):
    """Animated card for processor selection"""
    
    clicked = Signal(str)  # Processor ID
    
    def __init__(self, processor_id: str, name: str, description: str, icon: str):
        super().__init__()
        self.processor_id = processor_id
        self.setFrameStyle(QFrame.Box)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(100)
        
        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Header with icon and name
        header_layout = QHBoxLayout()
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 24px;")
        header_layout.addWidget(icon_label)
        
        name_label = QLabel(name)
        name_label.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(name_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #888;")
        layout.addWidget(desc_label)
        
        self.setLayout(layout)
        
        # Hover animation
        self._opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(0.8)
        
        # Style
        self.setStyleSheet("""
            ProcessorCard {
                background-color: #2a2e32;
                border: 2px solid #3a3e42;
                border-radius: 8px;
            }
            ProcessorCard:hover {
                border-color: #3daee9;
                background-color: #2e3236;
            }
        """)
        
    def mousePressEvent(self, event):
        """Handle click with animation"""
        logger.debug(f"ProcessorCard clicked! ID: {self.processor_id}")
        if event.button() == Qt.LeftButton:
            logger.debug(f"Left click detected for {self.processor_id}")
            # Click animation
            self._click_animation = QPropertyAnimation(self._opacity_effect, b"opacity")
            self._click_animation.setDuration(200)
            self._click_animation.setKeyValueAt(0, 0.8)
            self._click_animation.setKeyValueAt(0.5, 0.5)
            self._click_animation.setKeyValueAt(1, 1.0)
            self._click_animation.start()
            
            logger.debug(f"Emitting clicked signal for {self.processor_id}")
            self.clicked.emit(self.processor_id)
            
    def enterEvent(self, event):
        """Animate on hover"""
        animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        animation.setDuration(200)
        animation.setEndValue(1.0)
        animation.start()
        
    def leaveEvent(self, event):
        """Animate on leave"""
        animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        animation.setDuration(200)
        animation.setEndValue(0.8)
        animation.start()


class ParameterWidget(QWidget):
    """Widget for displaying and editing processing parameters"""
    
    value_changed = Signal(str, str, Any)  # category, param_name, value
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__()
        self.parameters = parameters
        self.widgets = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Create parameter controls"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create groups for each category
        for category, params in self.parameters.items():
            group = QGroupBox(category.replace('_', ' ').title())
            group_layout = QVBoxLayout()
            
            for param_name, param_info in params.items():
                param_layout = QHBoxLayout()
                
                # Label
                label = QLabel(param_info['description'])
                label.setWordWrap(True)
                label.setMinimumWidth(150)
                param_layout.addWidget(label)
                
                # Create appropriate widget based on type
                widget = self._create_param_widget(param_info)
                self.widgets[f"{category}.{param_name}"] = widget
                param_layout.addWidget(widget)
                
                # Connect change signal
                self._connect_widget_signal(widget, category, param_name)
                
                group_layout.addLayout(param_layout)
                
            group.setLayout(group_layout)
            layout.addWidget(group)
            
        layout.addStretch()
        self.setLayout(layout)
        
    def _create_param_widget(self, param_info: Dict[str, Any]) -> QWidget:
        """Create appropriate widget for parameter type"""
        param_type = param_info.get('type', 'float')
        value = param_info.get('value', 0)
        
        if param_type == 'bool':
            widget = QCheckBox()
            widget.setChecked(value)
        elif param_type == 'int':
            widget = QSpinBox()
            widget.setMinimum(param_info.get('min', 0))
            widget.setMaximum(param_info.get('max', 100))
            widget.setValue(value)
        elif param_type == 'float':
            widget = QDoubleSpinBox()
            widget.setMinimum(param_info.get('min', 0.0))
            widget.setMaximum(param_info.get('max', 100.0))
            widget.setDecimals(3)
            widget.setValue(value)
        elif param_type == 'choice':
            widget = QComboBox()
            widget.addItems(param_info.get('choices', []))
            widget.setCurrentText(str(value))
        elif param_type == 'file':
            # Create file selection widget
            widget = self._create_file_widget(param_info, value)
        else:
            widget = QLabel(str(value))
            
        return widget
        
    def _create_file_widget(self, param_info: Dict[str, Any], value: str) -> QWidget:
        """Create a file selection widget with browse button"""
        file_widget = QWidget()
        file_layout = QHBoxLayout()
        file_layout.setContentsMargins(0, 0, 0, 0)
        
        # Line edit for file path
        file_line_edit = QLineEdit()
        file_line_edit.setText(str(value) if value else "")
        file_line_edit.setPlaceholderText("Select file...")
        file_line_edit.setMinimumWidth(200)
        file_layout.addWidget(file_line_edit)
        
        # Browse button
        browse_button = QPushButton("Browse...")
        browse_button.setMaximumWidth(80)
        
        # Get file types for filter
        file_types = param_info.get('file_types', ['*'])
        if file_types and file_types != ['*']:
            filter_str = f"Supported files ({' '.join(f'*{ext}' for ext in file_types)});;All files (*.*)"
        else:
            filter_str = "All files (*.*)"
        
        # Store file types and filter in widget for access in browse function
        file_widget._file_types = file_types
        file_widget._filter_str = filter_str
        file_widget._line_edit = file_line_edit
        
        # Connect browse button - use a proper method reference
        browse_button.clicked.connect(lambda checked, fw=file_widget: self._browse_file(fw))
        file_layout.addWidget(browse_button)
        
        file_widget.setLayout(file_layout)
        file_widget.file_line_edit = file_line_edit  # Store reference for value retrieval
        
        return file_widget
    
    def _browse_file(self, file_widget: QWidget):
        """Handle file browsing for file widgets"""
        try:
            logger.debug(f"Opening file browser with filter: {file_widget._filter_str}")
            
            # Get the parent window for the dialog
            parent_window = self
            while parent_window.parent():
                parent_window = parent_window.parent()
            
            logger.debug(f"Using parent window: {parent_window}")
            
            file_path, selected_filter = QFileDialog.getOpenFileName(
                parent_window,
                "Select Base Station File", 
                "", 
                file_widget._filter_str
            )
            
            logger.debug(f"File dialog returned: path='{file_path}', filter='{selected_filter}'")
            
            if file_path:
                file_widget._line_edit.setText(file_path)
                logger.info(f"Selected file: {file_path}")
            else:
                logger.debug("File dialog was cancelled or no file selected")
                
        except Exception as e:
            logger.error(f"Error in file browser: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
    def _connect_widget_signal(self, widget: QWidget, category: str, param_name: str):
        """Connect widget change signal"""
        if isinstance(widget, QCheckBox):
            widget.toggled.connect(
                lambda val: self.value_changed.emit(category, param_name, val)
            )
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(
                lambda val: self.value_changed.emit(category, param_name, val)
            )
        elif isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(
                lambda val: self.value_changed.emit(category, param_name, val)
            )
        elif hasattr(widget, 'file_line_edit'):  # File widget
            widget.file_line_edit.textChanged.connect(
                lambda val: self.value_changed.emit(category, param_name, val)
            )
            
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values"""
        result = {}
        for category, params in self.parameters.items():
            result[category] = {}
            for param_name, param_info in params.items():
                widget_key = f"{category}.{param_name}"
                widget = self.widgets.get(widget_key)
                
                if isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    value = widget.value()
                elif isinstance(widget, QComboBox):
                    value = widget.currentText()
                elif hasattr(widget, 'file_line_edit'):  # File widget
                    value = widget.file_line_edit.text()
                else:
                    value = param_info.get('value')
                    
                result[category][param_name] = {'value': value}
                
        return result


class ProcessingWidget(QWidget):
    """Main processing widget with processor selection and execution"""
    
    # Signals
    processing_complete = Signal(ProcessingResult)
    layer_created = Signal(object)  # UXOLayer created during processing
    # NOTE: data_updated signal removed for clean processor architecture
    # data_updated = Signal(pd.DataFrame)
    
    def __init__(self, project_manager=None):
        super().__init__()
        self.pipeline = ProcessingPipeline(project_manager)
        self.current_data: Optional[pd.DataFrame] = None
        self.current_input_file: Optional[str] = None
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Initialize the UI with animations"""
        layout = QVBoxLayout()
        
        # Title with fade-in animation
        title_label = QLabel("🚀 Data Processing Center")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title_label)
        
        # Stacked widget for different views
        self.stacked_widget = QStackedWidget()
        
        # View 1: Processor selection
        self.selection_widget = self._create_selection_view()
        self.stacked_widget.addWidget(self.selection_widget)
        
        # View 2: Processing view
        self.processing_widget = self._create_processing_view()
        self.stacked_widget.addWidget(self.processing_widget)
        
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)
        
        # Start with selection view
        self.show_selection_view()
        
    def _create_selection_view(self) -> QWidget:
        """Create processor selection view with cards"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel("Select a processor based on your data type:")
        info_label.setStyleSheet("padding: 10px; color: #888;")
        layout.addWidget(info_label)
        
        # Processor cards in a grid
        cards_widget = QWidget()
        cards_layout = QVBoxLayout()
        
        processors = self.pipeline.get_available_processors()
        logger.debug(f"Found {len(processors)} processors")
        
        for proc_info in processors:
            card = ProcessorCard(
                proc_info['id'],
                proc_info['name'],
                proc_info['description'],
                proc_info['icon']
            )
            card.clicked.connect(self.select_processor)
            cards_layout.addWidget(card)
            
        cards_widget.setLayout(cards_layout)
        
        # Scroll area for cards
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(cards_widget)
        layout.addWidget(scroll_area)
        
        # Auto-detect button
        auto_button = QPushButton("🎯 Auto-Detect Data Type")
        auto_button.clicked.connect(self.auto_detect_processor)
        layout.addWidget(auto_button)
        
        widget.setLayout(layout)
        return widget
        
    def _create_processing_view(self) -> QWidget:
        """Create processing execution view"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Back button
        back_button = QPushButton("← Back to Selection")
        back_button.clicked.connect(self.show_selection_view)
        layout.addWidget(back_button)
        
        # Processor info
        self.processor_label = QLabel()
        self.processor_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.processor_label)
        
        # Parameters scroll area
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        layout.addWidget(self.params_scroll)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready to process")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = AnimatedProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.process_button = QPushButton("⚡ Start Processing")
        self.process_button.clicked.connect(self.start_processing)
        button_layout.addWidget(self.process_button)
        
        self.cancel_button = QPushButton("✖ Cancel")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        widget.setLayout(layout)
        return widget
        
    def connect_signals(self):
        """Connect pipeline signals"""
        self.pipeline.processing_started.connect(self.on_processing_started)
        self.pipeline.processing_finished.connect(self.on_processing_finished)
        self.pipeline.progress_updated.connect(self.on_progress_updated)
        self.pipeline.error_occurred.connect(self.on_error)
        
        # Forward layer creation signal from pipeline to widget signal
        self.pipeline.layer_created.connect(self.layer_created.emit)
        
    def set_data(self, data: pd.DataFrame, input_file_path: Optional[str] = None):
        """Set data to process"""
        self.current_data = data
        self.current_input_file = input_file_path
        self.current_processor = None  # Track current processor for script switching
        self._updating_parameters = False  # Flag to prevent recursive parameter updates
        logger.info(f"Data loaded: {len(data)} rows, {len(data.columns)} columns")
        if input_file_path:
            logger.info(f"Input file: {input_file_path}")
        
    def show_selection_view(self):
        """Show processor selection"""
        logger.debug("Switching to selection view")
        self.stacked_widget.setCurrentIndex(0)
        
    def show_processing_view(self):
        """Show processing view"""
        logger.debug("Switching to processing view")
        self.stacked_widget.setCurrentIndex(1)
        logger.debug(f"Current widget index: {self.stacked_widget.currentIndex()}")
        logger.debug(f"Current widget: {self.stacked_widget.currentWidget()}")
        
    def _animate_view_change(self, index: int):
        """Simple view transition without animation"""
        logger.debug(f"Animating view change to index {index}")
        self.stacked_widget.setCurrentIndex(index)
        
    def select_processor(self, processor_id: str):
        """Select a processor and show its parameters"""
        logger.debug(f"select_processor called with ID: {processor_id}")
        processor = self.pipeline.get_processor(processor_id)
        logger.debug(f"Got processor: {processor}")
        if not processor:
            logger.warning(f"No processor found for ID: {processor_id}")
            return
            
        # Update label
        self.processor_label.setText(f"{processor.name}")
        logger.debug(f"Set processor label to: {processor.name}")
        
        # Create parameter widget
        logger.debug(f"Creating parameter widget with {len(processor.parameters)} parameter groups")
        self.params_widget = ParameterWidget(processor.parameters)
        self.params_widget.value_changed.connect(self.on_parameter_changed)
        self.params_scroll.setWidget(self.params_widget)
        logger.debug("Created parameter widget and set it to scroll area")
        
        # Store current processor for script switching
        self.current_processor = processor
        
        # Show processing view
        logger.debug("Showing processing view")
        self.show_processing_view()
        
    def auto_detect_processor(self):
        """Auto-detect appropriate processor"""
        if self.current_data is None:
            logger.warning("Auto-detect requested but no data loaded")
            self.on_error("No data loaded")
            return
            
        processor_id = self.pipeline.detect_data_type(self.current_data)
        logger.info(f"Auto-detected processor: {processor_id}")
        self.select_processor(processor_id)
    
    def on_parameter_changed(self, category: str, param_name: str, value: Any):
        """Handle parameter changes, particularly script selection"""
        logger.debug(f"Parameter changed: {category}.{param_name} = {value}")
        
        # Prevent recursive parameter updates
        if self._updating_parameters:
            logger.debug("Skipping parameter change during update")
            return
        
        # Handle script selection changes
        if category == 'script_selection' and param_name == 'script_name' and self.current_processor:
            logger.info(f"Script changed to: {value}")
            
            try:
                # Set flag to prevent recursive updates
                self._updating_parameters = True
                
                # Update processor's current script and regenerate parameters
                logger.debug(f"Setting script to: {value}")
                self.current_processor.set_script(value)
                logger.debug(f"Script parameters: {len(self.current_processor.parameters)} categories")
                
                # Safely remove old widget first
                old_widget = self.params_scroll.widget()
                if old_widget:
                    # Disconnect all signals first to prevent crashes
                    try:
                        # Disconnect the value_changed signal if it exists
                        if hasattr(old_widget, 'value_changed'):
                            old_widget.value_changed.disconnect()
                    except:
                        pass  # Ignore if already disconnected
                    
                    # Remove from scroll area
                    self.params_scroll.takeWidget()
                    # Set parent to None and schedule for deletion
                    old_widget.setParent(None)
                    old_widget.deleteLater()
                    
                # Process events to ensure cleanup happens
                QCoreApplication.processEvents()
                
                # Create new parameter widget with new script parameters
                logger.debug("Creating new ParameterWidget")
                new_params_widget = ParameterWidget(self.current_processor.parameters)
                logger.debug("Connecting value_changed signal")
                new_params_widget.value_changed.connect(self.on_parameter_changed)
                
                # Set the new widget
                logger.debug("Setting widget to scroll area")
                self.params_scroll.setWidget(new_params_widget)
                self.params_widget = new_params_widget
                
                logger.debug("Parameter widget recreated with new script parameters")
                
            except Exception as e:
                logger.error(f"Error switching scripts: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                # Always reset the flag
                self._updating_parameters = False
        
    def start_processing(self):
        """Start processing with current parameters"""
        if self.current_data is None:
            self.on_error("No data to process")
            return
            
        # Get parameters
        params = self.params_widget.get_parameters()
        
        # Get current processor ID from label
        processor_name = self.processor_label.text()
        processor_id = None
        for proc_info in self.pipeline.get_available_processors():
            if proc_info['name'] == processor_name:
                processor_id = proc_info['id']
                break
                
        if not processor_id:
            return
            
        # Start processing
        logger.info(f"DEBUG PROCESSING WIDGET: About to call pipeline.process_data with processor_id={processor_id}")
        logger.info(f"DEBUG PROCESSING WIDGET: current_data shape={self.current_data.shape if self.current_data is not None else 'None'}")
        self.pipeline.process_data(processor_id, self.current_data, params, self.current_input_file)
        logger.info(f"DEBUG PROCESSING WIDGET: pipeline.process_data call completed")
        
    def cancel_processing(self):
        """Cancel current processing"""
        self.pipeline.cancel_processing()
        
    def on_processing_started(self, processor_name: str):
        """Handle processing start"""
        self.process_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.set_indeterminate(True)
        self.status_label.setText(f"Processing with {processor_name}...")
        
    def on_processing_finished(self, result: ProcessingResult):
        """Handle processing completion"""
        self.process_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.set_indeterminate(False)
        
        if result.success:
            self.progress_bar.setValue(100)
            self.status_label.setText("Processing complete!")
            self.processing_complete.emit(result)
            
            # NOTE: Automatic data signal emission removed for clean processor architecture
            # if result.data is not None:
            #     self.data_updated.emit(result.data)
                
            # Show success animation
            self._show_success_animation()
        else:
            self.status_label.setText(f"Error: {result.error_message}")
            self.progress_bar.setValue(0)
            
    def on_progress_updated(self, value: int, message: str):
        """Update progress display"""
        if value >= 0:
            self.progress_bar.set_indeterminate(False)
            self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
            
    def on_error(self, error_message: str):
        """Handle processing error"""
        self.status_label.setText(f"Error: {error_message}")
        self.process_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        logger.error(error_message)
        
    def _show_success_animation(self):
        """Show success animation on progress bar"""
        # Flash effect
        effect = QGraphicsOpacityEffect()
        self.progress_bar.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(1000)
        animation.setKeyValueAt(0, 1.0)
        animation.setKeyValueAt(0.5, 0.3)
        animation.setKeyValueAt(1, 1.0)
        animation.setLoopCount(2)
        animation.start() 