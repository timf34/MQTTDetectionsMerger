import json
import os
import threading
import logging
import random
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


def generate_random_detection(camera_id):
    return Detections(
        camera_id=camera_id,
        probability=round(random.uniform(0.8, 1.0), 2),
        timestamp=time(),
        x=random.randint(0, 1920),
        y=random.randint(0, 1080),
        z=round(random.uniform(0.5, 1.5), 2)
    )


def send_hardcoded_detections():
    try:
        while True:
            detections = [
                asdict(generate_random_detection(camera_id=1)),
                asdict(generate_random_detection(camera_id=2)),
                asdict(generate_random_detection(camera_id=3)),
                asdict(generate_random_detection(camera_id=5)),
                asdict(generate_random_detection(camera_id=6))
            ]

            mqtt_message = {
                "message": detections,
                "time": time()
            }

            iot_manager.publish(payload=json.dumps(mqtt_message))

            # Sleep for a random interval between 0.01 and 0.5 seconds
            sleep_interval = random.uniform(0.01, 0.5)
            print(f"Sleeping for {sleep_interval} seconds")
            sleep(sleep_interval)

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        iot_manager.disconnect()
        print("IOT Client disconnected")


if __name__ == "__main__":
    send_hardcoded_detections()
