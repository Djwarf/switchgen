"""SwitchGen entry point."""

import sys


def main():
    """Main entry point for SwitchGen."""
    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        from .core.test_headless import run_test
        run_test()
        return

    # Normal GTK4 application launch
    from .app import run_app
    run_app()


if __name__ == "__main__":
    main()
