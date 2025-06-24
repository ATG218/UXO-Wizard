"""
Theme Manager for UXO Wizard - Handle application theming
"""

from PySide6.QtCore import QFile, QTextStream
from PySide6.QtWidgets import QApplication
from loguru import logger


class ThemeManager:
    """Manages application themes"""
    
    def __init__(self):
        self.themes = {
            "dark": self.get_dark_theme(),
            "light": self.get_light_theme()
        }
        self.current_theme = "dark"
        
    def apply_theme(self, widget, theme_name):
        """Apply a theme to a widget"""
        if theme_name in self.themes:
            self.current_theme = theme_name
            widget.setStyleSheet(self.themes[theme_name])
            logger.info(f"Applied {theme_name} theme")
        else:
            logger.warning(f"Theme not found: {theme_name}")
            
    def get_dark_theme(self):
        """Dark theme stylesheet"""
        return """
        /* Dark Theme for UXO Wizard */
        
        /* Main Window */
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        /* Widgets */
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        
        /* Tab Widget */
        QTabWidget::pane {
            border: 1px solid #3c3c3c;
            background-color: #2b2b2b;
        }
        
        QTabBar::tab {
            background-color: #3c3c3c;
            color: #ffffff;
            padding: 8px 16px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background-color: #0d7377;
            color: #ffffff;
        }
        
        QTabBar::tab:hover {
            background-color: #4a4a4a;
        }
        
        /* Dock Widgets */
        QDockWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            titlebar-close-icon: url(close.png);
            titlebar-normal-icon: url(float.png);
        }
        
        QDockWidget::title {
            background-color: #3c3c3c;
            padding: 6px;
            border: 1px solid #3c3c3c;
        }
        
        /* Menu Bar */
        QMenuBar {
            background-color: #2b2b2b;
            color: #ffffff;
            border-bottom: 1px solid #3c3c3c;
        }
        
        QMenuBar::item:selected {
            background-color: #0d7377;
        }
        
        QMenu {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #3c3c3c;
        }
        
        QMenu::item:selected {
            background-color: #0d7377;
        }
        
        /* Tool Bar */
        QToolBar {
            background-color: #2b2b2b;
            border: none;
            padding: 4px;
        }
        
        QToolButton {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #3c3c3c;
            padding: 4px 8px;
            margin: 2px;
        }
        
        QToolButton:hover {
            background-color: #4a4a4a;
            border: 1px solid #0d7377;
        }
        
        QToolButton:pressed {
            background-color: #0d7377;
        }
        
        QToolButton:checked {
            background-color: #0d7377;
            border: 1px solid #14ffec;
        }
        
        /* Status Bar */
        QStatusBar {
            background-color: #2b2b2b;
            color: #ffffff;
            border-top: 1px solid #3c3c3c;
        }
        
        /* Tree View */
        QTreeView {
            background-color: #2b2b2b;
            color: #ffffff;
            selection-background-color: #0d7377;
            border: 1px solid #3c3c3c;
            outline: none;
        }
        
        QTreeView::item:hover {
            background-color: #3c3c3c;
        }
        
        QTreeView::item:selected {
            background-color: #0d7377;
        }
        
        /* Table View */
        QTableView {
            background-color: #2b2b2b;
            color: #ffffff;
            gridline-color: #3c3c3c;
            selection-background-color: #0d7377;
            border: 1px solid #3c3c3c;
        }
        
        QHeaderView::section {
            background-color: #3c3c3c;
            color: #ffffff;
            padding: 6px;
            border: none;
            border-right: 1px solid #2b2b2b;
            border-bottom: 1px solid #2b2b2b;
        }
        
        /* Text Edit */
        QTextEdit {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #3c3c3c;
            selection-background-color: #0d7377;
        }
        
        /* Line Edit */
        QLineEdit {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #4a4a4a;
            padding: 4px;
            selection-background-color: #0d7377;
        }
        
        QLineEdit:focus {
            border: 1px solid #0d7377;
        }
        
        /* Combo Box */
        QComboBox {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #4a4a4a;
            padding: 4px;
            min-width: 80px;
        }
        
        QComboBox:hover {
            border: 1px solid #0d7377;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox::down-arrow {
            image: url(down_arrow.png);
            width: 12px;
            height: 12px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #2b2b2b;
            color: #ffffff;
            selection-background-color: #0d7377;
            border: 1px solid #3c3c3c;
        }
        
        /* Progress Bar */
        QProgressBar {
            background-color: #3c3c3c;
            border: 1px solid #4a4a4a;
            text-align: center;
            color: #ffffff;
        }
        
        QProgressBar::chunk {
            background-color: #0d7377;
        }
        
        /* Slider */
        QSlider::groove:horizontal {
            background-color: #3c3c3c;
            height: 6px;
            border-radius: 3px;
        }
        
        QSlider::handle:horizontal {
            background-color: #0d7377;
            width: 16px;
            height: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        
        QSlider::handle:horizontal:hover {
            background-color: #14ffec;
        }
        
        /* Scroll Bar */
        QScrollBar:vertical {
            background-color: #2b2b2b;
            width: 12px;
            border: none;
        }
        
        QScrollBar::handle:vertical {
            background-color: #4a4a4a;
            min-height: 20px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #5a5a5a;
        }
        
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        /* Message Box */
        QMessageBox {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        
        QPushButton {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #4a4a4a;
            padding: 6px 16px;
            min-width: 80px;
        }
        
        QPushButton:hover {
            background-color: #4a4a4a;
            border: 1px solid #0d7377;
        }
        
        QPushButton:pressed {
            background-color: #0d7377;
        }
        
        /* Label */
        QLabel {
            color: #ffffff;
            background-color: transparent;
        }
        """
        
    def get_light_theme(self):
        """Light theme stylesheet"""
        return """
        /* Light Theme for UXO Wizard */
        
        /* Main Window */
        QMainWindow {
            background-color: #f5f5f5;
            color: #333333;
        }
        
        /* Widgets */
        QWidget {
            background-color: #ffffff;
            color: #333333;
        }
        
        /* Tab Widget */
        QTabWidget::pane {
            border: 1px solid #d0d0d0;
            background-color: #ffffff;
        }
        
        QTabBar::tab {
            background-color: #e0e0e0;
            color: #333333;
            padding: 8px 16px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background-color: #1976d2;
            color: #ffffff;
        }
        
        QTabBar::tab:hover {
            background-color: #c0c0c0;
        }
        
        /* Dock Widgets */
        QDockWidget {
            background-color: #ffffff;
            color: #333333;
        }
        
        QDockWidget::title {
            background-color: #e0e0e0;
            padding: 6px;
            border: 1px solid #d0d0d0;
        }
        
        /* Menu Bar */
        QMenuBar {
            background-color: #ffffff;
            color: #333333;
            border-bottom: 1px solid #d0d0d0;
        }
        
        QMenuBar::item:selected {
            background-color: #1976d2;
            color: #ffffff;
        }
        
        QMenu {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #d0d0d0;
        }
        
        QMenu::item:selected {
            background-color: #1976d2;
            color: #ffffff;
        }
        
        /* Tool Bar */
        QToolBar {
            background-color: #f5f5f5;
            border: none;
            padding: 4px;
        }
        
        QToolButton {
            background-color: #e0e0e0;
            color: #333333;
            border: 1px solid #d0d0d0;
            padding: 4px 8px;
            margin: 2px;
        }
        
        QToolButton:hover {
            background-color: #c0c0c0;
            border: 1px solid #1976d2;
        }
        
        QToolButton:pressed {
            background-color: #1976d2;
            color: #ffffff;
        }
        
        QToolButton:checked {
            background-color: #1976d2;
            color: #ffffff;
            border: 1px solid #0d47a1;
        }
        
        /* Status Bar */
        QStatusBar {
            background-color: #f5f5f5;
            color: #333333;
            border-top: 1px solid #d0d0d0;
        }
        
        /* Tree View */
        QTreeView {
            background-color: #ffffff;
            color: #333333;
            selection-background-color: #1976d2;
            selection-color: #ffffff;
            border: 1px solid #d0d0d0;
            outline: none;
        }
        
        QTreeView::item:hover {
            background-color: #e3f2fd;
        }
        
        QTreeView::item:selected {
            background-color: #1976d2;
            color: #ffffff;
        }
        
        /* Table View */
        QTableView {
            background-color: #ffffff;
            color: #333333;
            gridline-color: #d0d0d0;
            selection-background-color: #1976d2;
            selection-color: #ffffff;
            border: 1px solid #d0d0d0;
        }
        
        QHeaderView::section {
            background-color: #e0e0e0;
            color: #333333;
            padding: 6px;
            border: none;
            border-right: 1px solid #d0d0d0;
            border-bottom: 1px solid #d0d0d0;
        }
        
        /* Text Edit */
        QTextEdit {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #d0d0d0;
            selection-background-color: #1976d2;
            selection-color: #ffffff;
        }
        
        /* Line Edit */
        QLineEdit {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #d0d0d0;
            padding: 4px;
            selection-background-color: #1976d2;
            selection-color: #ffffff;
        }
        
        QLineEdit:focus {
            border: 1px solid #1976d2;
        }
        
        /* Combo Box */
        QComboBox {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #d0d0d0;
            padding: 4px;
            min-width: 80px;
        }
        
        QComboBox:hover {
            border: 1px solid #1976d2;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #333333;
            selection-background-color: #1976d2;
            selection-color: #ffffff;
            border: 1px solid #d0d0d0;
        }
        
        /* Progress Bar */
        QProgressBar {
            background-color: #e0e0e0;
            border: 1px solid #d0d0d0;
            text-align: center;
            color: #333333;
        }
        
        QProgressBar::chunk {
            background-color: #1976d2;
        }
        
        /* Slider */
        QSlider::groove:horizontal {
            background-color: #d0d0d0;
            height: 6px;
            border-radius: 3px;
        }
        
        QSlider::handle:horizontal {
            background-color: #1976d2;
            width: 16px;
            height: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        
        QSlider::handle:horizontal:hover {
            background-color: #0d47a1;
        }
        
        /* Scroll Bar */
        QScrollBar:vertical {
            background-color: #f5f5f5;
            width: 12px;
            border: none;
        }
        
        QScrollBar::handle:vertical {
            background-color: #c0c0c0;
            min-height: 20px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #a0a0a0;
        }
        
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        /* Message Box */
        QMessageBox {
            background-color: #ffffff;
            color: #333333;
        }
        
        QPushButton {
            background-color: #e0e0e0;
            color: #333333;
            border: 1px solid #d0d0d0;
            padding: 6px 16px;
            min-width: 80px;
        }
        
        QPushButton:hover {
            background-color: #c0c0c0;
            border: 1px solid #1976d2;
        }
        
        QPushButton:pressed {
            background-color: #1976d2;
            color: #ffffff;
        }
        
        /* Label */
        QLabel {
            color: #333333;
            background-color: transparent;
        }
        """ 