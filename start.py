import os
import subprocess
import sys

PORT = int(os.environ.get("PORT", "8899"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=PORT, reload=False)