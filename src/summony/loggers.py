from abc import abstractmethod
import logging
import uuid
from pathlib import Path
import time
import datetime
import traceback
import json


logger = logging.getLogger(__name__)


def make_default_logger(
    name: str | None = None, level: int = logging.DEBUG, file_path: str | None = None
):
    if name is None:
        name = f"logger-{uuid.uuid4().hex[:8]}"

    logger = logging.Logger(name)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter("%(levelname)s %(asctime)s %(message)s @%(name)s")

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if file_path is not None:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


class XLoggerInterface:
    # --- logging.Logger methods
    @abstractmethod
    def log(self, level: int, msg, *args, **kwargs): ...

    @abstractmethod
    def debug(self, msg: str, *args, **kwargs) -> None: ...

    @abstractmethod
    def info(self, msg: str, *args, **kwargs) -> None: ...

    @abstractmethod
    def warning(self, msg: str, *args, **kwargs) -> None: ...

    @abstractmethod
    def error(self, msg: str, *args, **kwargs) -> None: ...

    @abstractmethod
    def exception(self, msg: str, *args, exc_info=True, **kwargs) -> None:
        self.error(msg, *args, **kwargs)

    @abstractmethod
    def critical(self, msg: str, *args, **kwargs) -> None: ...

    @abstractmethod
    def fatal(self, msg: str, *args, **kwargs) -> None: ...

    def log_model_call(
        self,
        *,
        req_content: dict,
        req_base_url: str | None = None,
        req_url: str | None = None,
        req_headers: dict | None = None,
        res_content: dict = None,
        res_status_code: int | None = None,
        res_headers: dict | None = None,
        error: Exception | None = None,
    ): ...

    def log_model_reply_chunk(self, chunk: dict, error: Exception | None = None): ...


class DefaultXLogger(XLoggerInterface):
    _name: str
    _logger: logging.Logger

    def __init__(
        self,
        logger: logging.Logger | None = None,
        model_logs_path: str | None = None,
        name: str | None = None,
    ):
        super().__init__()

        # timesatamp int + random 8-char hex
        suffix = str(int(time.time() * 1000)) + "-" + uuid.uuid4().hex[:4]
        if logger is not None:
            self._logger = logger
        else:
            logs_path = Path(__file__).parent.resolve() / "logs"
            logs_path.mkdir(parents=True, exist_ok=True)
            self._logger = make_default_logger(
                file_path=logs_path / "log.log", name=f"logger-{name}-{suffix}"
            )

        self._name = self._logger.name if name is None else name

        if model_logs_path is None:
            model_logs_path = (
                Path(__file__).parent.resolve()
                / "logs"
                / f"agent-{self._name}-{suffix}"
            )
        model_logs_path = Path(model_logs_path)
        model_logs_path.mkdir(parents=True, exist_ok=True)

        self._model_logs_path = model_logs_path

        self._logger = logger if logger is not None else make_default_logger()

    def log(self, level: int, msg, *args, **kwargs):
        self._logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg: str, *args, exc_info=True, **kwargs) -> None:
        self.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def fatal(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def log_model_call(
        self,
        *,
        req_content: dict,
        req_base_url: str | None = None,
        req_url: str | None = None,
        req_headers: dict | None = None,
        res_content: dict = None,
        res_status_code: int | None = None,
        res_headers: dict | None = None,
        error: Exception | None = None,
    ):
        now = datetime.datetime.now(datetime.timezone.utc)
        filename = (
            now.strftime("%Y-%m-%d_%H-%M-%S-%f_") + uuid.uuid4().hex[:8] + ".json"
        )
        log_path = Path(self._model_logs_path).name + "/" + filename
        with open(self._model_logs_path / filename, "w") as f:
            req = {}
            if req_base_url is not None:
                req["request_base_url"] = req_base_url
            if req_url is not None:
                req["request_url"] = req_url
            if req_headers is not None:
                req["request_headers"] = req_headers
            req.update(req_content)

            res = {}
            if res_status_code is not None:
                res["response_status_code"] = res_status_code
            if res_headers is not None:
                res["response_headers"] = res_headers
            if res_content is not None:
                res.update(res_content)

            err = {}
            if error is not None:
                err["error"] = str(error)
                if stacktrace := traceback.format_exc():
                    err["stacktrace"] = stacktrace

            to_log = {"request": req}
            if res:
                to_log["response"] = res
            if err:
                to_log["error"] = err

            json.dump(to_log, f, ensure_ascii=True, indent=2)
        return log_path

    def log_model_reply_chunk(self, chunk: dict): ...
