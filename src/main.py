from .app import run_application, build_application
from .config.loader import Config
from .utils.logger import setup_logging
import logging

logger = logging.getLogger(__name__)

def main():
    setup_logging()
    config = Config()
    controller = build_application(config=config)
    run_application(controller=controller)
    
if __name__ == "__main__":
    main()
