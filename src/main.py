import os

import logging

import uvicorn

from .api.app import create_app
from .utils.logger import setup_logging

logger = logging.getLogger(__name__)

def main() -> None:
    setup_logging()
    uvicorn.run(
        create_app(),
        host=os.getenv("WEB_HOST", "0.0.0.0"),
        port=int(os.getenv("WEB_PORT", "8080")),
        log_config=None,
    )
    
if __name__ == "__main__":
    main()
