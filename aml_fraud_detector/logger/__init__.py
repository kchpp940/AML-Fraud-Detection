import logging
import os
from datetime import datetime
from typing import Optional


_default_log_file_name = None
_default_log_dir = "logs"
_configured = False


def _get_default_log_file_name() -> str:
    global _default_log_file_name
    if _default_log_file_name is None:
        _default_log_file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    return _default_log_file_name


def _get_default_logs_path() -> str:
    log_file = _get_default_log_file_name()
    logs_path = os.path.join(os.getcwd(), _default_log_dir, log_file)
    os.makedirs(logs_path, exist_ok=True)
    return os.path.join(logs_path, log_file)


def configure_logging(
    logs_dir: Optional[str] = None,
    log_file_name: Optional[str] = None,
    level: int = logging.DEBUG,
    reset: bool = False,
) -> str:
    """
    Configure the logging system with a custom logs directory.

    Args:
        logs_dir: Directory to store log files. If None, uses default.
        log_file_name: Name of the log file. If None, auto-generates one.
        level: Logging level (default: DEBUG).
        reset: If True, removes all existing handlers before reconfiguring.

    Returns:
        The absolute path of the log file.
    """
    global _configured

    if logs_dir:
        log_dir_abs = os.path.abspath(logs_dir)
    else:
        log_dir_abs = os.path.join(os.getcwd(), _default_log_dir)

    if log_file_name:
        log_file = log_file_name
    else:
        log_file = _get_default_log_file_name()

    log_subdir = os.path.join(log_dir_abs, log_file)
    os.makedirs(log_subdir, exist_ok=True)
    log_file_path = os.path.join(log_subdir, log_file)

    root_logger = logging.getLogger()

    if reset or not _configured:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        root_logger.setLevel(level)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        _configured = True
        logging.info(f"Logging configured: {log_file_path}")

    return log_file_path


def configure_logging_from_workspace(workspace) -> str:
    """
    Configure logging using a WorkspaceContext.

    Args:
        workspace: A WorkspaceContext instance.

    Returns:
        The absolute path of the log file.
    """
    return configure_logging(
        logs_dir=workspace.logs_dir,
        log_file_name=_get_default_log_file_name(),
        reset=True,
    )


if not _configured:
    LOG_FILE = _get_default_log_file_name()
    logs_path = _get_default_logs_path()
    LOG_FILE_PATH = logs_path

    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    _configured = True
