import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from gui.main_window import ImageEditor
import logging
from config import get_logs_dir, ICON_PATHS

log_file = os.path.join(get_logs_dir(), "app.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def main():
    logging.info("Starting application...")
    app = QApplication(sys.argv)
    
    # Set application icon
    if os.path.exists(ICON_PATHS['app_icon']):
        app.setWindowIcon(QIcon(ICON_PATHS['app_icon']))
    
    editor = ImageEditor()
    editor.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()