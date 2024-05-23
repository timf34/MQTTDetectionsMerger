import json
import os
import threading
import logging

from dataclasses import asdict
from logging import Formatter, FileHandler
from time import time, sleep

from aws_iot.IOTClient import IOTClient
from aws_iot.IOTContext import IOTContext, IOTCredentials
from config import MQTTMergerConfig
from data_models import Detections
from triangulation.triangulation_logic import MultiCameraTracker
from utils import convert_dicts_to_detections

config = MQTTMergerConfig()

iot_context = IOTContext()
iot_credentials = IOTCredentials(
    cert_path=config.cert_path,
    client_id="mergerReceiveMessages",
    endpoint=config.endpoint,
    region="eu-west-1",
    priv_key_path=config.private_key_path,
    ca_path=config.root_ca_path
)

# IOT Manager receive.
iot_manager = IOTClient(iot_context, iot_credentials, subscribe_topic=config.camera_topic, publish_topic=config.device_topic)
connect_future = iot_manager.connect()
print("IOT receive manager connected!")


detections_store = {
    "1": [asdict(Detections(camera_id=1, probability=0, timestamp=0, x=100, y=100))],
    "2": [asdict(Detections(camera_id=2, probability=0, timestamp=0, x=200, y=200))],
    "3": [asdict(Detections(camera_id=3, probability=0, timestamp=0, x=300, y=300))],
    "5": [asdict(Detections(camera_id=5, probability=0, timestamp=0, x=500, y=500))],
    "6": [asdict(Detections(camera_id=6, probability=0, timestamp=0, x=600, y=600))],
}


while True:
    pass
    # mqtt_message = {
    #     "message": detections,
    #     "time": time.time()
    # }
    #
    # iot_manager.publish(payload=json.dumps(mqtt_message))





