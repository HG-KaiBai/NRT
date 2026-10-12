"""Atomic CSV replacement with retries for transient Windows reader locks."""
import time


def write_csv(frame,path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary,index=False)
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05*2**min(attempt,4))
