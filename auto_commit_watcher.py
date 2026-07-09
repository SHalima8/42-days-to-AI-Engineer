import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

REPO_PATH = "."
COOLDOWN_SECONDS = 3

class GitAutoCommitHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_run = 0

    def on_modified(self, event):
        if event.is_directory or ".git" in event.src_path:
            return

        now = time.time()
        if now - self.last_run < COOLDOWN_SECONDS:
            return  # ignore duplicate/rapid-fire events
        self.last_run = now

        subprocess.run(["git", "add", "."], cwd=REPO_PATH)
        result = subprocess.run(
            ["git", "commit", "-m", f"auto commit: {time.strftime('%Y-%m-%d %H:%M:%S')}"],
            cwd=REPO_PATH, capture_output=True, text=True
        )
        subprocess.run(["git", "push"], cwd=REPO_PATH)
        print(f"✅ Committed and pushed at {time.strftime('%H:%M:%S')}")

observer = Observer()
observer.schedule(GitAutoCommitHandler(), path=REPO_PATH, recursive=True)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()