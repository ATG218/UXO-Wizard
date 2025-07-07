#!/usr/bin/env python
"""
UXO Wizard Desktop Suite - Main Entry Point
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from loguru import logger

def main():
    """Main application entry point"""
    # Configure logging
    logger.add("uxo_wizard.log", rotation="10 MB", level="DEBUG")
    logger.info("Starting UXO Wizard Desktop Suite")
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("UXO Wizard Desktop Suite")
    app.setOrganizationName("UXO-Wizard")
    
    # Set application icon
    icon_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "src/ui/assets/icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        logger.warning(f"Icon not found at {icon_path}, using default.")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 