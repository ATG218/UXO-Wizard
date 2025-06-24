#!/usr/bin/env python
"""
UXO Wizard Desktop Suite - Main Entry Point
"""

import sys
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
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 