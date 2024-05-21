import json
import os

from logging import Formatter, FileHandler
from typing import Dict, List


class JsonFormatter(Formatter):
    def format(self, record):
        # Create a structured log message as a JSON object
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_record)