
"""
Plot Widget for UXO Wizard - Display matplotlib figures (2D and 3D)
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PySide6.QtCore import QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # Enable 3D plotting support
from loguru import logger

class PlotWidget(QWidget):
    """A widget to display a matplotlib figure (supports 2D and 3D plots)"""
    def __init__(self, figure: Figure = None, parent=None):
        super().__init__(parent)
        # Create a default figure if none is provided.
        self.figure = figure if figure is not None else Figure()
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI for this widget"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        
        # Add a toolbar for navigation with smaller icons
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(self.toolbar.iconSize() * 0.2)  # Reduce icon size by 30%
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def set_figure(self, figure: Figure):
        """Set or update the figure to display"""
        self.figure = figure
        
        # Store references to old widgets before removal
        old_canvas = getattr(self, 'canvas', None)
        old_toolbar = getattr(self, 'toolbar', None)
        
        # Create new canvas and toolbar with the new figure
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(self.toolbar.iconSize() * 0.7)  # Reduce icon size by 30%
        
        # Clear the old layout safely
        layout = self.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().hide()  # Hide first to prevent drawing issues
        
        # Add new widgets to layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Draw the new canvas
        self.canvas.draw()
        
        # Clean up old widgets after a delay to avoid Qt issues
        if old_canvas or old_toolbar:
            QTimer.singleShot(100, lambda: self._cleanup_old_widgets(old_canvas, old_toolbar))
        
        logger.debug("New figure set and drawn in PlotWidget")
    
    def _cleanup_old_widgets(self, old_canvas, old_toolbar):
        """Safely cleanup old widgets after a delay"""
        try:
            if old_canvas:
                old_canvas.deleteLater()
            if old_toolbar:
                old_toolbar.deleteLater()
        except RuntimeError:
            # Widget already deleted, ignore
            pass

    def get_figure(self) -> Figure:
        """Get the current figure"""
        return self.figure
