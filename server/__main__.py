"""``python -m server`` -- run the batch API with uvicorn."""

import logging
import os

import uvicorn


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    uvicorn.run(
        "server.app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        workers=1,  # one process: the job queues and warm ONNX sessions are in-process
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        timeout_keep_alive=int(os.environ.get("KEEP_ALIVE", "30")),
    )


if __name__ == "__main__":
    main()
