import os
import platform


def notify_complete() -> None:
    """Notify when cell execution completes"""
    if platform.system() == "Darwin":  # macOS
        os.system("afplay /System/Library/Sounds/Glass.aiff")
    else:  # Other platforms
        print("\a")  # ASCII bell
