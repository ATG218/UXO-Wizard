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
            "light": self.get_light_theme(),
            "cyberpunk": self.get_cyberpunk_theme()
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
        
    def get_cyberpunk_theme(self):
        """Cyberpunk theme stylesheet - NEON FUTURE AESTHETIC"""
        return """
        /* 🌆 CYBERPUNK THEME - NEON DREAMS 🌆 */
        
        /* Main Window - The Matrix */
        QMainWindow {
            background-color: #0a0a0f;
            color: #00ffff;
            border: 1px solid #ff00ff;
        }
        
        /* Base Widgets - Dark Matrix */
        QWidget {
            background-color: #0f0f1a;
            color: #00ffff;
            font-family: "Consolas", "Monaco", "Courier New", monospace;
        }
        
        /* Tab Widget - Neon Panels */
        QTabWidget::pane {
            border: 2px solid #ff00ff;
            background-color: #0f0f1a;
            border-radius: 5px;
        }
        
        QTabBar::tab {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #1a1a2e, stop: 1 #16213e);
            color: #00ffff;
            padding: 10px 20px;
            margin-right: 3px;
            border: 1px solid #ff00ff;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: bold;
            min-width: 100px;
        }
        
        QTabBar::tab:selected {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #ff00ff, stop: 1 #8000ff);
            color: #ffffff;
            border-bottom: 3px solid #00ffff;
            box-shadow: 0 0 20px #ff00ff;
        }
        
        QTabBar::tab:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #00ffff, stop: 1 #0080ff);
            color: #000000;
            border: 1px solid #00ffff;
            box-shadow: 0 0 15px #00ffff;
        }
        
        /* Dock Widgets - Holographic Panels */
        QDockWidget {
            background-color: #0a0a0f;
            color: #00ffff;
            border: 2px solid #ff00ff;
            border-radius: 8px;
        }
        
        QDockWidget::title {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #ff00ff, stop: 0.5 #8000ff, stop: 1 #00ffff);
            padding: 8px;
            border-radius: 5px;
            color: #ffffff;
            font-weight: bold;
            text-align: center;
        }
        
        /* Menu Bar - Neon Strip */
        QMenuBar {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #1a1a2e, stop: 1 #0f0f1a);
            color: #00ffff;
            border-bottom: 3px solid #ff00ff;
            padding: 4px;
            font-weight: bold;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 8px 16px;
            border-radius: 4px;
        }
        
        QMenuBar::item:selected {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #ff00ff, stop: 1 #8000ff);
            color: #ffffff;
            border: 1px solid #00ffff;
            box-shadow: 0 0 10px #ff00ff;
        }
        
        QMenu {
            background-color: #0f0f1a;
            color: #00ffff;
            border: 2px solid #ff00ff;
            border-radius: 6px;
            padding: 4px;
        }
        
        QMenu::item {
            padding: 8px 20px;
            border-radius: 4px;
        }
        
        QMenu::item:selected {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #ff00ff, stop: 1 #00ffff);
            color: #ffffff;
            border: 1px solid #ffffff;
            box-shadow: 0 0 8px #ff00ff;
        }
        
        /* Tool Bar - Command Console */
        QToolBar {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #1a1a2e, stop: 1 #0f0f1a);
            border: 2px solid #ff00ff;
            padding: 6px;
            border-radius: 8px;
            spacing: 4px;
        }
        
        QToolButton {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #2a2a3e, stop: 1 #1a1a2e);
            color: #00ffff;
            border: 2px solid #ff00ff;
            padding: 8px 12px;
            margin: 2px;
            border-radius: 6px;
            font-weight: bold;
            min-width: 60px;
        }
        
        QToolButton:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #00ffff, stop: 1 #0080ff);
            color: #000000;
            border: 2px solid #ffffff;
            box-shadow: 0 0 15px #00ffff;
        }
        
        QToolButton:pressed {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #ff00ff, stop: 1 #8000ff);
            color: #ffffff;
            border: 2px solid #00ffff;
            box-shadow: 0 0 20px #ff00ff;
        }
        
        QToolButton:checked {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #ff00ff, stop: 1 #00ffff);
            color: #ffffff;
            border: 2px solid #ffffff;
            box-shadow: 0 0 25px #ff00ff;
        }
        
        /* Status Bar - Data Stream */
        QStatusBar {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #0f0f1a, stop: 0.5 #1a1a2e, stop: 1 #0f0f1a);
            color: #00ffff;
            border-top: 3px solid #ff00ff;
            padding: 4px;
            font-family: "Consolas", monospace;
            font-weight: bold;
        }
        
        /* Tree View - Data Matrix */
        QTreeView {
            background-color: #0f0f1a;
            color: #00ffff;
            selection-background-color: #ff00ff;
            selection-color: #ffffff;
            border: 2px solid #ff00ff;
            border-radius: 6px;
            outline: none;
            font-family: "Consolas", monospace;
        }
        
        QTreeView::item {
            padding: 4px;
            border-bottom: 1px solid #2a2a3e;
        }
        
        QTreeView::item:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #00ffff, stop: 1 #0080ff);
            color: #000000;
            border: 1px solid #ffffff;
            border-radius: 3px;
        }
        
        QTreeView::item:selected {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #ff00ff, stop: 1 #8000ff);
            color: #ffffff;
            border: 1px solid #00ffff;
            border-radius: 3px;
            font-weight: bold;
        }
        
        QTreeView::branch:has-children:!has-siblings:closed,
        QTreeView::branch:closed:has-children:has-siblings {
            border-image: none;
            image: none;
        }
        
        QTreeView::branch:open:has-children:!has-siblings,
        QTreeView::branch:open:has-children:has-siblings {
            border-image: none;
            image: none;
        }
        
        /* Table View - Holographic Data Grid */
        QTableView {
            background-color: #0f0f1a;
            color: #00ffff;
            gridline-color: #ff00ff;
            selection-background-color: #ff00ff;
            selection-color: #ffffff;
            border: 2px solid #ff00ff;
            border-radius: 6px;
            font-family: "Consolas", monospace;
        }
        
        QTableView::item {
            padding: 4px;
            border-bottom: 1px solid #2a2a3e;
        }
        
        QTableView::item:hover {
            background-color: #00ffff;
            color: #000000;
        }
        
        QTableView::item:selected {
            background-color: #ff00ff;
            color: #ffffff;
            font-weight: bold;
        }
        
        QHeaderView::section {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #ff00ff, stop: 1 #8000ff);
            color: #ffffff;
            padding: 8px;
            border: 1px solid #00ffff;
            font-weight: bold;
            text-align: center;
        }
        
        /* Text Edit - Terminal Screen */
        QTextEdit {
            background-color: #0a0a0f;
            color: #00ff00;
            border: 2px solid #00ffff;
            border-radius: 6px;
            selection-background-color: #ff00ff;
            selection-color: #ffffff;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 11pt;
            padding: 8px;
        }
        
        /* Line Edit - Input Terminal */
        QLineEdit {
            background-color: #1a1a2e;
            color: #00ffff;
            border: 2px solid #ff00ff;
            border-radius: 6px;
            padding: 8px;
            selection-background-color: #ff00ff;
            selection-color: #ffffff;
            font-family: "Consolas", monospace;
            font-weight: bold;
        }
        
        QLineEdit:focus {
            border: 2px solid #00ffff;
            background-color: #2a2a3e;
            box-shadow: 0 0 15px #00ffff;
        }
        
        /* Combo Box - Selection Pod */
        QComboBox {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #2a2a3e, stop: 1 #1a1a2e);
            color: #00ffff;
            border: 2px solid #ff00ff;
            border-radius: 6px;
            padding: 6px 12px;
            min-width: 100px;
            font-weight: bold;
        }
        
        QComboBox:hover {
            border: 2px solid #00ffff;
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #00ffff, stop: 1 #0080ff);
            color: #000000;
            box-shadow: 0 0 10px #00ffff;
        }
        
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 25px;
            border-left: 1px solid #ff00ff;
            border-radius: 3px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #0f0f1a;
            color: #00ffff;
            selection-background-color: #ff00ff;
            selection-color: #ffffff;
            border: 2px solid #ff00ff;
            border-radius: 6px;
        }
        
        /* Progress Bar - Data Transfer */
        QProgressBar {
            background-color: #1a1a2e;
            border: 2px solid #ff00ff;
            border-radius: 8px;
            text-align: center;
            color: #ffffff;
            font-weight: bold;
            padding: 2px;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #00ffff, stop: 0.5 #ff00ff, stop: 1 #8000ff);
            border-radius: 6px;
            box-shadow: 0 0 10px #ff00ff;
        }
        
        /* Slider - Control Interface */
        QSlider::groove:horizontal {
            background-color: #2a2a3e;
            height: 8px;
            border-radius: 4px;
            border: 1px solid #ff00ff;
        }
        
        QSlider::handle:horizontal {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #ff00ff, stop: 1 #00ffff);
            width: 20px;
            height: 20px;
            margin: -6px 0;
            border-radius: 10px;
            border: 2px solid #ffffff;
        }
        
        QSlider::handle:horizontal:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                       stop: 0 #00ffff, stop: 1 #ff00ff);
            box-shadow: 0 0 15px #ff00ff;
        }
        
        /* Scroll Bar - Interface Chrome */
        QScrollBar:vertical {
            background-color: #1a1a2e;
            width: 16px;
            border: 1px solid #ff00ff;
            border-radius: 8px;
        }
        
        QScrollBar::handle:vertical {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #ff00ff, stop: 1 #00ffff);
            min-height: 30px;
            border-radius: 7px;
            border: 1px solid #ffffff;
        }
        
        QScrollBar::handle:vertical:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #00ffff, stop: 1 #ff00ff);
            box-shadow: 0 0 10px #ff00ff;
        }
        
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        /* Message Box - Alert Protocol */
        QMessageBox {
            background-color: #0f0f1a;
            color: #00ffff;
            border: 3px solid #ff00ff;
            border-radius: 10px;
        }
        
        /* Push Button - Action Triggers */
        QPushButton {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #2a2a3e, stop: 1 #1a1a2e);
            color: #00ffff;
            border: 2px solid #ff00ff;
            border-radius: 8px;
            padding: 10px 20px;
            min-width: 100px;
            font-weight: bold;
            font-size: 10pt;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #00ffff, stop: 1 #0080ff);
            color: #000000;
            border: 2px solid #ffffff;
            box-shadow: 0 0 20px #00ffff;
        }
        
        QPushButton:pressed {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #ff00ff, stop: 1 #8000ff);
            color: #ffffff;
            border: 2px solid #00ffff;
            box-shadow: 0 0 25px #ff00ff;
        }
        
        /* Label - Data Display */
        QLabel {
            color: #00ffff;
            background-color: transparent;
            font-weight: bold;
        }
        
        /* Special Effects for Status Elements */
        QStatusBar QLabel {
            color: #00ff00;
            font-family: "Consolas", monospace;
            font-weight: bold;
        }
        
        /* Console/Terminal Special Styling */
        QTextEdit[objectName="console"] {
            background-color: #000000;
            color: #00ff00;
            border: 2px solid #00ff00;
            font-family: "Consolas", "Monaco", "Courier New", monospace;
            font-size: 10pt;
        }
        """ 