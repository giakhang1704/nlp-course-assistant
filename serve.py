"""Local, single-worker web entry point. python -B serve.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("course_assistant.web.app:create_app", factory=True, host="127.0.0.1", port=8000)
