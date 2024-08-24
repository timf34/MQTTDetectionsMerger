import json
import numpy as np
import os
import threading
import logging
from copy import deepcopy
from dataclasses import asdict
from logging import Formatter, FileHandler
from time import time, sleep
from typing import Optional, Dict, List, Union

from aws_iot.IOTClient import IOTClient
from aws_iot.IOTContext import IOTContext, IOTCredentials
from config import MQTTMergerConfig
from data_models import Detections
from round_trip_latency_measuring import MQTTLatencyMeasurer
from triangulation.triangulation_logic import MultiCameraTracker
from utils import convert_dicts_to_detections

# TODO: generally needs to ensure the timing mechanisms in send_detections_periodically is correct


NUM_MESSAGES = 10000000
received_count = 0
elapsed_time = 0
received_message: str = ""
received_all_event = threading.Event()
detections_buffer = {}
flow_vector_buffer: Dict[int, Optional[np.ndarray]] = {}

config = MQTTMergerConfig()

tracker = MultiCameraTracker(
    sport="afl",
    camera_coords_json_path="./data/afl_camera_coordinates.json"
)

start_time = time()

# Configure the logger
logger = logging.getLogger("detection_logger")
logger.setLevel(logging.INFO)

# Create a file handler
cwd = os.getcwd()
log_file = os.path.join(cwd, "detections_test2.log")
file_handler = FileHandler(log_file)
file_handler.setLevel(logging.INFO)


# Create and set a JSON formatter
class JsonFormatter(Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.msg
        }
        return json.dumps(log_record, indent=2)


json_formatter = JsonFormatter()
file_handler.setFormatter(json_formatter)

# Add the handler to the logger
logger.addHandler(file_handler)


def log_incoming_detections(received_message):
    """
    Logs the incoming detections when received.
    """
    log_entry = {
        "type": "incoming_detections",
        "message": received_message,
    }
    logger.info(log_entry)


def log_buffered_detections(detections_to_send):
    """
    Logs detections that make it through the buffer.
    """
    log_entry = {
        "type": "buffered_detections",
        "detections": [d.__dict__ for d in detections_to_send],
    }
    logger.info(log_entry)


def log_3d_detections(mqtt_message, three_d_point):
    """
    Logs the 3D detections that are sent to the device.
    """
    log_message = {
        "mqtt_message": mqtt_message,
        "three_d_point": asdict(three_d_point),
    }
    log_entry = {
        "type": "3d_detections",
        "message": log_message,
    }
    logger.info(log_entry)


def on_message_received(topic, payload, dup, qos, retain, **kwargs):
    global received_count, received_message, elapsed_time, detections_buffer

    received_count += 1
    received_message = payload

    end_time = time()
    elapsed_time = end_time - start_time

    if isinstance(received_message, bytes) and received_message != '':
        received_message = received_message.decode("utf-8")

    # # Log the received message and timestamp
    # log_entry = {
    #     "type": "received",
    #     "message": received_message,
    #     "timestamp": elapsed_time
    # }
    # logger.info(json.dumps(log_entry))

    received_message_json = json.loads(received_message)
    log_incoming_detections(received_message_json)

    # if received_message_json.get("detections") != []:
    #     print("detections: ", received_message_json.get("detections"))


    detection_list = received_message_json["detections"]
    detection_list = convert_dicts_to_detections(detection_list)

    for detection in detection_list:
        # TODO: this is a bit of a mistake... the last detection then is the one that will be stored in the buffer,
        #  and not the most confident one
        #  Come back to this
        detections_buffer[detection.camera_id] = detection

    if received_count == NUM_MESSAGES:
        received_all_event.set()


# TODO: as this is in a thread, errors don't stop things!
def send_detections_periodically():
    print(received_all_event.is_set())
    while not received_all_event.is_set():
        current_time = time()
        detections_to_send = []

        for camera_id, detection in detections_buffer.items():
            # TODO: Change this back to 0.6 seconds for production
            if current_time - detection.timestamp <= 0.6:
                detections_to_send.append(detection)

        # print("\n buffer: ", detections_to_send)
        # print(current_time)

        if detections_to_send:

            log_buffered_detections(detections_to_send)

            # Dets in the detections buffer get adjusted sometimes if we don't use deepcopy here
            three_d_point = tracker.multi_camera_analysis(deepcopy(detections_to_send), {})
            # print("three d point: ", three_d_point)

            if three_d_point is not None:

                # Converting to normalized coordinates for the device
                normalized_x = min(three_d_point.x, 159.5)
                normalized_y = min(three_d_point.y, 128.8)

                normalized_x = normalized_x / 159.5
                normalized_x *= 102

                normalized_y = 128.8 - normalized_y  # Keep commented out for correct visualisation on laptop at least
                normalized_y = normalized_y / 128.8
                normalized_y *= 65

                mqtt_message = {
                  "T": 0,
                  "X": int(normalized_x),
                  "Y": int(normalized_y),
                  "P": 1,
                  "Pa": 0,
                  "G": 0,
                  "O": 0
                }
                log_3d_detections(mqtt_message, three_d_point)
                iot_manager.publish(payload=json.dumps(mqtt_message))
            else:
                tracker.increment_frame_count_since_two_camera_det()

        sleep(1 / 6)  # Wait for 1/6 seconds


if __name__ == "__main__":
    print("Change the detections buffer back to 0.4 seconds for production!")
    sleep(3)

    cwd = os.getcwd()

    iot_context = IOTContext()

    iot_credentials = IOTCredentials(
        cert_path=config.cert_path,
        client_id="mergerReceiveMessagesxxx",
        endpoint=config.endpoint,
        region="ap-southeast-2",
        priv_key_path=config.private_key_path,
        ca_path=config.root_ca_path
    )

    # IOT Manager receive.
    iot_manager = IOTClient(iot_context, iot_credentials, subscribe_topic=config.camera_topic, publish_topic=config.device_topic)
    connect_future = iot_manager.connect()
    print("IOT receive manager connected!")

    subscribe_future = iot_manager.subscribe(topic=iot_manager.subscribe_topic, handler=on_message_received)
    subscribe_result = subscribe_future.result()
    print(f"Subscribed with {str(subscribe_result['qos'])}")

    if not received_all_event.is_set():
        print("Waiting to receive message.")

    # Start the periodic sending in a separate thread
    threading.Thread(target=send_detections_periodically, daemon=True).start()

    # Set a timeout for 6 hours (3 hours * 60 minutes/hour * 60 seconds/minute)

    latency_sender = MQTTLatencyMeasurer()

    # TODO: Ensure that this can run indefinitely and doesn't time out
    temp_received_count = 0
    while not received_all_event.is_set():
        if len(received_message) == 0:
            print("received_message is empty. Continuing...")
            sleep(0.5)
            continue


    latency_sender.start()  # Only start once we have received a message at least
    received_all_event.wait()  # https://docs.python.org/3/library/threading.html#threading.Event.wait used with .set()

    latency_sender.stop()
    disconnect_future = iot_manager.disconnect()
    disconnect_future.result()
    print("Disconnected!")
