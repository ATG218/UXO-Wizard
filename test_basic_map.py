#!/usr/bin/env python3
"""
Working PySide6 + pyqtlet2 demo.
"""

import os, sys

# 1) Make sure qtpy and your own code use **the same** Qt binding.
os.environ["QT_API"] = "pyside6"          # or "pyqt5" if you prefer

from qtpy.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget
)
from qtpy.QtWebEngineCore import QWebEngineSettings
from pyqtlet2 import L, MapWidget


class BasicMap(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyqtlet2 – basic map")
        self.resize(800, 600)

        # Central widget + layout
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Leaflet container
        self.map_widget = MapWidget()

        # 2) Let local HTML load remote (tile) URLs
        self.map_widget.settings().setAttribute(
            QWebEngineSettings.LocalContentCanAccessRemoteUrls, True
        )

        layout.addWidget(self.map_widget)

        # 3) Build the Leaflet map (no QTimer needed – MapWidget waits for loadFinished)
        self.map = L.map(self.map_widget)
        self.map.setView([59.9139, 10.7522], 10)

        L.tileLayer(
         "https://cache.kartverket.no/v1/wmts/1.0.0/topo/default/webmercator/{z}/{y}/{x}.png",
            {"attribution": "© Kartverket", 
            "maxZoom": 18,          
            "tileSize": 256
            }
        ).addTo(self.map)


        marker = L.marker([59.9139, 10.7522])
        marker.bindPopup("Oslo, Norway")
        marker.addTo(self.map)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = BasicMap()
    win.show()
    sys.exit(app.exec())
