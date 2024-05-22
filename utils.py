import json
import os

from logging import Formatter, FileHandler
from typing import Dict, List

from data_models import Detections


class JsonFormatter(Formatter):
    def format(self, record):
        # Create a structured log message as a JSON object
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_record)


def convert_dicts_to_detections(dicts: List[Dict]) -> List[Detections]:
    return [Detections(**d) for d in dicts]
