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
    QTimer, Property, QCoreApplication, QSettings
)
from PySide6.QtGui import QPalette, QFont
import pandas as pd
from typing import Dict, Any, Optional, List
from loguru import logger
import json
from datetime import datetime

from ....processing import ProcessingPipeline, ProcessingResult
from ....processing.base import ScriptMetadata


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
        elif param_type == 'directory':
            # Create directory selection widget
            widget = self._create_directory_widget(param_info, value)
        else:
            widget = QLabel(str(value))
            
        return widget
        
    def _create_directory_widget(self, param_info: Dict[str, Any], value: str) -> QWidget:
        """Create a directory selection widget with a browse button."""
        dir_widget = QWidget()
        dir_layout = QHBoxLayout()
        dir_layout.setContentsMargins(0, 0, 0, 0)

        # Line edit for directory path
        dir_line_edit = QLineEdit()
        dir_line_edit.setText(str(value) if value else "")
        dir_line_edit.setPlaceholderText("Select directory...")
        dir_line_edit.setMinimumWidth(200)
        dir_layout.addWidget(dir_line_edit)

        # Browse button
        browse_button = QPushButton("Browse...")
        browse_button.setMaximumWidth(80)

        dir_widget._line_edit = dir_line_edit
        browse_button.clicked.connect(lambda: self._browse_directory(dir_widget))
        dir_layout.addWidget(browse_button)

        dir_widget.setLayout(dir_layout)
        dir_widget.dir_line_edit = dir_line_edit  # For value retrieval
        return dir_widget

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
    
    def _browse_directory(self, dir_widget: QWidget):
        """Handle directory browsing for directory widgets."""
        try:
            logger.debug("Opening directory browser.")
            parent_window = self.parent()
            while parent_window.parent():
                parent_window = parent_window.parent()

            dir_path = QFileDialog.getExistingDirectory(
                parent_window,
                "Select Directory",
                dir_widget._line_edit.text() or "",
            )

            if dir_path:
                dir_widget._line_edit.setText(dir_path)
                logger.info(f"Selected directory: {dir_path}")
            else:
                logger.debug("Directory dialog was cancelled.")
        except Exception as e:
            logger.error(f"Error in directory browser: {e}", exc_info=True)

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
                lambda val: self.value_changed.emit(category, param_name, widget.currentData() or val)
            )
        elif hasattr(widget, 'file_line_edit'):  # File widget
            widget.file_line_edit.textChanged.connect(
                lambda val: self.value_changed.emit(category, param_name, val)
            )
        elif hasattr(widget, 'dir_line_edit'):  # Directory widget
            widget.dir_line_edit.textChanged.connect(
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
                elif hasattr(widget, 'dir_line_edit'): # Directory widget
                    value = widget.dir_line_edit.text()
                else:
                    value = param_info.get('value')
                    
                result[category][param_name] = {'value': value}
                
        return result


class ProcessingWidget(QWidget):
    """Main processing widget with processor selection and execution"""
    
    # Class variable to track last used script across all instances
    _session_last_used = {}  # {"magnetic": {"script": "path_visualize", "params": {...}}, ...}
    
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
        
    @classmethod
    def set_last_used_script(cls, processor_type: str, script_name: str, parameters: dict):
        """Track the last used script and parameters for this session"""
        cls._session_last_used[processor_type] = {
            "script": script_name,
            "parameters": parameters.copy(),
            "timestamp": datetime.now()
        }

    @classmethod  
    def get_last_used_script(cls, processor_type: str = None):
        """Get the most recently used script, optionally filtered by processor type"""
        if processor_type:
            return cls._session_last_used.get(processor_type)
        
        # Return most recent across all processor types
        if not cls._session_last_used:
            return None
        
        most_recent = None
        latest_time = None
        
        for proc_type, info in cls._session_last_used.items():
            if latest_time is None or info["timestamp"] > latest_time:
                latest_time = info["timestamp"]
                most_recent = {
                    "processor_type": proc_type,
                    **info
                }
        
        return most_recent
        
    def _get_recent_scripts(self, processor_type: str) -> List[str]:
        """Get recent scripts from QSettings"""
        settings = QSettings("UXO-Wizard", "Desktop-Suite")
        recent_json = settings.value(f"recent_scripts/{processor_type}", "[]")
        try:
            return json.loads(recent_json)
        except:
            return []

    def _save_recent_script(self, processor_type: str, script_name: str):
        """Save script to recent list (max 5 items)"""
        settings = QSettings("UXO-Wizard", "Desktop-Suite")
        recent = self._get_recent_scripts(processor_type)
        
        # Remove if already exists
        if script_name in recent:
            recent.remove(script_name)
        
        # Add to front
        recent.insert(0, script_name)
        
        # Limit to 5 items
        recent = recent[:5]
        
        # Save back to settings
        settings.setValue(f"recent_scripts/{processor_type}", json.dumps(recent))

    def _format_script_tooltip(self, metadata: ScriptMetadata) -> str:
        """Format script metadata into a rich tooltip"""
        flags_str = " • ".join([f"#{flag}" for flag in metadata.flags]) if metadata.flags else "No tags"
        
        tooltip_parts = [
            f"<b>{metadata.description}</b>",
            "",
            f"<b>Tags:</b> {flags_str}",
            f"<b>Runtime:</b> {metadata.estimated_runtime}",
            f"<b>Field Compatible:</b> {'Yes' if metadata.field_compatible else 'No'}",
            "",
            f"<i>{metadata.typical_use_case}</i>"
        ]
        
        return "<br>".join(tooltip_parts)
        
    def _enhance_script_dropdown(self, processor, set_default_selection=False):
        """Enhance script dropdown with tooltips and recent prioritization"""
        if not self.params_widget or not processor:
            return
            
        # Set flag to prevent recursive parameter updates during enhancement
        was_updating = getattr(self, '_updating_parameters', False)
        self._updating_parameters = True
            
        # Find the script selection combo box in the parameter widget
        script_combo = None
        all_combos = self.params_widget.findChildren(QComboBox)
        
        for combo in all_combos:
            # Check if this combo is for script selection by examining its parent hierarchy
            parent = combo
            found_script_selection = False
            
            # Walk up the parent hierarchy looking for script_selection group
            for _ in range(10):  # Limit depth to prevent infinite loops
                parent = parent.parent()
                if not parent:
                    break
                # Check if parent is a QGroupBox with "Script Selection" in title
                if hasattr(parent, 'title') and 'script' in parent.title().lower():
                    found_script_selection = True
                    break
            
            if found_script_selection:
                script_combo = combo
                break
        
        # Also try to find it by checking the combo box items
        if not script_combo:
            for combo in all_combos:
                if combo.count() > 0:
                    # Check if any items look like script names
                    first_item = combo.itemText(0)
                    if any(script_word in first_item.lower() for script_word in ['process', 'analyz', 'visual', 'grid', 'anomaly']):
                        script_combo = combo
                        break
                
        if not script_combo:
            return
            
        # Get recent scripts for this processor type
        recent_scripts = self._get_recent_scripts(processor.processor_type)
        
        # Get scripts with recent priority if the method exists
        if hasattr(processor, 'get_scripts_with_recent_priority'):
            scripts = processor.get_scripts_with_recent_priority(recent_scripts)
        else:
            # Fallback: get all scripts and manually prioritize recent ones
            all_scripts = processor.get_available_scripts() if hasattr(processor, 'get_available_scripts') else {}
            scripts = {}
            
            # Add recent scripts first
            for script_name in recent_scripts:
                if script_name in all_scripts:
                    scripts[script_name] = all_scripts[script_name]
            
            # Add remaining scripts
            for script_name, script_instance in all_scripts.items():
                if script_name not in scripts:
                    scripts[script_name] = script_instance
        
        # Store current selection before clearing
        current_selection = script_combo.currentData() or script_combo.currentText()
        
        # Clear and repopulate combo box
        script_combo.clear()
        
        recent_count = 0
        for script_name, script_instance in scripts.items():
            # Add script to dropdown (no star, just ordered by recency then alphabetical)
            if script_name in recent_scripts:
                recent_count += 1
            
            script_combo.addItem(script_name, script_name)
            
            # Get metadata and create rich tooltip if method exists
            if hasattr(processor, 'get_script_metadata'):
                metadata = processor.get_script_metadata(script_name)
                if metadata:
                    tooltip = self._format_script_tooltip(metadata)
                    script_combo.setItemData(
                        script_combo.count() - 1, 
                        tooltip, 
                        Qt.ToolTipRole
                    )
        
        # Handle selection based on whether we're setting default or preserving user choice
        if set_default_selection and script_combo.count() > 0:
            # Set the first item (most recent script) as default when initially creating the widget
            self._updating_parameters = was_updating
            script_combo.setCurrentIndex(0)
            self._updating_parameters = True
        elif not set_default_selection and current_selection and script_combo.count() > 0:
            # Restore the previous selection when just enhancing (not setting default)
            for i in range(script_combo.count()):
                if script_combo.itemData(i) == current_selection or script_combo.itemText(i) == current_selection:
                    self._updating_parameters = was_updating
                    script_combo.setCurrentIndex(i)
                    self._updating_parameters = True
                    break
        
        
        # Restore the updating flag to its previous state
        self._updating_parameters = was_updating
        
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
        
        # Enhance script dropdown with tooltips, recent prioritization, and set default selection
        self._enhance_script_dropdown(processor, set_default_selection=True)
        
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
                
                # Enhance script dropdown with tooltips and recent prioritization (but don't override user selection)
                self._enhance_script_dropdown(self.current_processor, set_default_selection=False)
                
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

        # Log processing start event
        if self.pipeline.project_manager and self.pipeline.project_manager.history_logger:
            script_id = params.get('script_selection', {}).get('script_name', {}).get('value')
            self.pipeline.project_manager.history_logger.log_event(
                'processing.started',
                {
                    'processor_id': processor_id,
                    'script_id': script_id,
                    'input_file': self.current_input_file,
                    'parameters': params
                }
            )
            
        # Start processing
        logger.info(f"DEBUG PROCESSING WIDGET: About to call pipeline.process_data with processor_id={processor_id}")
        logger.info(f"DEBUG PROCESSING WIDGET: current_data shape={self.current_data.shape if self.current_data is not None else 'None'}")
        self.pipeline.process_data(processor_id, self.current_data, params, self.current_input_file)
        logger.info(f"DEBUG PROCESSING WIDGET: pipeline.process_data call completed")
        
        # Track script usage for session and recent scripts
        script_id = params.get('script_selection', {}).get('script_name', {}).get('value')
        if script_id and self.current_processor:
            self.set_last_used_script(self.current_processor.processor_type, script_id, params)
            self._save_recent_script(self.current_processor.processor_type, script_id)
        
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

            # Log processing completion event
            if self.pipeline.project_manager and self.pipeline.project_manager.history_logger:
                outputs = []
                
                # Capture layer outputs
                if result.layer_outputs:
                    for layer_output in result.layer_outputs:
                        outputs.append({
                            'type': 'layer',
                            'layer_name': getattr(layer_output, 'name', 'Unnamed Layer'),
                            'layer_type': layer_output.layer_type,
                            'source_file': layer_output.data if isinstance(layer_output.data, str) else f"in-memory {type(layer_output.data).__name__}"
                        })
                
                # Capture plot outputs
                if hasattr(result, 'plot_outputs') and result.plot_outputs:
                    for plot_output in result.plot_outputs:
                        outputs.append({
                            'type': 'plot',
                            'title': getattr(plot_output, 'title', 'Unnamed Plot'),
                            'source_file': getattr(plot_output, 'file_path', 'in-memory plot')
                        })
                
                # Capture data outputs  
                if hasattr(result, 'data_outputs') and result.data_outputs:
                    for data_output in result.data_outputs:
                        outputs.append({
                            'type': 'data',
                            'description': getattr(data_output, 'description', 'Processed data'),
                            'source_file': getattr(data_output, 'file_path', 'in-memory data')
                        })
                
                # Create comprehensive metadata dump including all ProcessingResult information
                metadata_dump = {
                    # Core processing result metadata (from script)
                    'script_metadata': result.metadata or {},
                    
                    # Processing execution details
                    'execution_details': {
                        'success': result.success,
                        'processing_time_seconds': result.processing_time,
                        'processing_script': result.processing_script,
                        'input_file_path': result.input_file_path,
                        'output_file_path': result.output_file_path,
                        'export_format': result.export_format,
                        'processor_type': result.processor_type,
                        'script_id': result.script_id,
                        'data_shape': list(result.data.shape) if result.data is not None else None,
                        'data_columns': list(result.data.columns) if result.data is not None else None
                    },
                    
                    # Output files information
                    'output_files': [
                        {
                            'file_path': of.file_path,
                            'file_type': of.file_type,
                            'description': of.description,
                            'metadata': of.metadata
                        } for of in result.output_files
                    ] if hasattr(result, 'output_files') else [],
                    
                    # Layer outputs information
                    'layer_outputs': [
                        {
                            'layer_type': lo.layer_type,
                            'data_type': type(lo.data).__name__,
                            'data_info': {
                                'shape': list(lo.data.shape) if hasattr(lo.data, 'shape') else None,
                                'columns': list(lo.data.columns) if hasattr(lo.data, 'columns') else None,
                                'length': len(lo.data) if hasattr(lo.data, '__len__') else None
                            },
                            'style_info': lo.style_info,
                            'metadata': lo.metadata
                        } for lo in result.layer_outputs
                    ] if hasattr(result, 'layer_outputs') else [],
                    
                    # Figure information
                    'figure_info': {
                        'has_figure': result.figure is not None,
                        'figure_type': type(result.figure).__name__ if result.figure else None
                    }
                }

                self.pipeline.project_manager.history_logger.log_event(
                    'processing.completed',
                    {
                        'status': 'success',
                        'execution_time_seconds': result.processing_time,
                        'error_message': None,
                        'outputs': outputs,
                        'metadata': metadata_dump  # Enhanced metadata dump
                    }
                )
            
            # NOTE: Automatic data signal emission removed for clean processor architecture
            # if result.data is not None:
            #     self.data_updated.emit(result.data)
                
            # Show success animation
            self._show_success_animation()
        else:
            self.status_label.setText(f"Error: {result.error_message}")
            self.progress_bar.setValue(0)
            
            # Log processing failure event
            if self.pipeline.project_manager and self.pipeline.project_manager.history_logger:
                # Create comprehensive metadata dump for failed processing too
                failure_metadata_dump = {
                    # Core processing result metadata (from script)
                    'script_metadata': result.metadata or {},
                    
                    # Processing execution details
                    'execution_details': {
                        'success': result.success,
                        'processing_time_seconds': result.processing_time,
                        'processing_script': result.processing_script,
                        'input_file_path': result.input_file_path,
                        'output_file_path': result.output_file_path,
                        'export_format': result.export_format,
                        'processor_type': result.processor_type,
                        'script_id': result.script_id,
                        'data_shape': list(result.data.shape) if result.data is not None else None,
                        'data_columns': list(result.data.columns) if result.data is not None else None,
                        'error_message': result.error_message
                    },
                    
                    # Output files information (may be empty for failures)
                    'output_files': [
                        {
                            'file_path': of.file_path,
                            'file_type': of.file_type,
                            'description': of.description,
                            'metadata': of.metadata
                        } for of in result.output_files
                    ] if hasattr(result, 'output_files') else [],
                    
                    # Layer outputs information (may be empty for failures)
                    'layer_outputs': [
                        {
                            'layer_type': lo.layer_type,
                            'data_type': type(lo.data).__name__,
                            'style_info': lo.style_info,
                            'metadata': lo.metadata
                        } for lo in result.layer_outputs
                    ] if hasattr(result, 'layer_outputs') else [],
                    
                    # Figure information
                    'figure_info': {
                        'has_figure': result.figure is not None,
                        'figure_type': type(result.figure).__name__ if result.figure else None
                    }
                }
                
                self.pipeline.project_manager.history_logger.log_event(
                    'processing.completed',
                    {
                        'status': 'failure',
                        'execution_time_seconds': result.processing_time,
                        'error_message': result.error_message,
                        'outputs': [],
                        'metadata': failure_metadata_dump  # Enhanced metadata dump
                    }
                )
            
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