"""SwitchGen entry point."""

import sys

from .core.logging import setup_logging, get_logger

# Initialize logging before anything else
setup_logging()
logger = get_logger(__name__)


def main():
    """Main entry point for SwitchGen."""
    logger.info("SwitchGen starting (Python %s)", sys.version.split()[0])

    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logger.info("Running in test mode")
        from .core.test_headless import run_test
        run_test()
        return

    # Normal GTK4 application launch
    logger.debug("Launching GTK4 application")
    from .app import run_app
    run_app()


if __name__ == "__main__":
    main()
