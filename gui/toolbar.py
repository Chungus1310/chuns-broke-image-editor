from PyQt5.QtWidgets import QToolBar, QAction
from PyQt5.QtCore import QSize

class ImageEditorToolBar(QToolBar):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setIconSize(QSize(16, 16))
        self.setMovable(False)
        self.setup_actions()

    def setup_actions(self):
        actions = [
            ("New", "📄", self.parent.new_image),
            ("Open", "📂", self.parent.open_image),
            ("Save", "💾", self.parent.save_image),
            (None, None, None),
            ("Undo", "↩", self.parent.undo),
            ("Redo", "↪", self.parent.redo),
            (None, None, None),
            ("Zoom In", "🔍+", self.parent.zoom_in),
            ("Zoom Out", "🔍-", self.parent.zoom_out),
            ("Fit Screen", "⤧", self.parent.fit_to_screen)
        ]
        
        for name, icon, func in actions:
            if name is None:
                self.addSeparator()
            else:
                action = QAction(f"{icon} {name}", self)
                action.triggered.connect(func)
                self.addAction(action)
# No changes required as toolbar does not handle model downloads