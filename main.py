"""Application entrypoint for the clustering visualizer."""

import sys

from PyQt5.QtWidgets import QApplication

from gui import MainWindow


def main() -> None:
    """Launch the Qt event loop and show the main window."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
