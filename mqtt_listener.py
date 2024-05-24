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
from triangulation.triangulation_logic import MultiCameraTracker
from utils import convert_dicts_to_detections

# TODO: Need to measure average latency between cameras and this MQTT channel. It's important for
# filtering out older detections (see below)


# TODO: Remove all the backslashes from the log messages.

NUM_MESSAGES = 10000000
received_count = 0
elapsed_time = 0
received_message: str = ""
received_all_event = threading.Event()
detections_buffer = {}

config = MQTTMergerConfig()

tracker = MultiCameraTracker(
    sport="afl",
    camera_coords_json_path="./triangulation/triangulation_data/afl_camera_coordinates.json"
)

start_time = time()

# Configure the logger
logger = logging.getLogger("detection_logger")
logger.setLevel(logging.INFO)

# Create a file handler
cwd = os.getcwd()
log_file = os.path.join(cwd, "detections.log")
file_handler = FileHandler(log_file)
file_handler.setLevel(logging.INFO)


# Create and set a JSON formatter
class JsonFormatter(Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_record)


json_formatter = JsonFormatter()
file_handler.setFormatter(json_formatter)

# Add the handler to the logger
logger.addHandler(file_handler)


def on_message_received(topic, payload, dup, qos, retain, **kwargs):
    global received_count, received_message, elapsed_time, detections_buffer

    print(f"Received message from topic '{topic}' at {time()}: {payload}")

    received_count += 1
    received_message = payload

    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Received {received_count} messages in {elapsed_time} seconds")

    if isinstance(received_message, bytes) and received_message != '':
        received_message = received_message.decode("utf-8")

    # Log the received message and timestamp
    log_entry = {
        "type": "received",
        "message": received_message,
        "timestamp": elapsed_time
    }
    logger.info(json.dumps(log_entry))

    received_message_json = json.loads(received_message)
    detection_list = received_message_json["message"]
    detection_list = convert_dicts_to_detections(detection_list)

    for detection in detection_list:
        detections_buffer[detection.camera_id] = detection

    if received_count == NUM_MESSAGES:
        received_all_event.set()


def send_detections_periodically():
    while not received_all_event.is_set():
        current_time = time()
        detections_to_send = []

        for camera_id, detection in detections_buffer.items():
            if current_time - detection.timestamp <= 0.35:  # Check if detection is within 1/4 second + 0.15 seconds for latency between cameras and server (I need to measure the average latency here for this)
                detections_to_send.append(detection)

        # if time() % 2 == 0:
        #     detections_to_send.append(
        #         Detections(camera_id=1, x=1027, y=844, z=1.0, probability=0.98, timestamp=time())
        #     )
        # else:
        #     detections_to_send.append(
        #         Detections(camera_id=6, x=964, y=801, z=1.0, probability=0.98, timestamp=time())
        # )

        if detections_to_send:
            log_entry = {
                "type": "detections_to_send",
                "detections": [d.__dict__ for d in detections_to_send],
                "timestamp": current_time
            }
            logger.info(json.dumps(log_entry))

            three_d_point = tracker.multi_camera_analysis(detections_to_send, {})
            if three_d_point is not None:

                # Converting to normalized coordinates for the device
                normalized_x = min(three_d_point.x, 159.5)
                normalized_y = min(three_d_point.y, 128.8)

                normalized_x = normalized_x / 159.5
                normalized_x *= 102

                normalized_y = normalized_y / 128.8
                normalized_y *= 128

                mqtt_message = {
                  "T": 0,
                  "X": int(normalized_x),
                  "Y": int(normalized_y),
                  "P": 1,
                  "Pa": 0,
                  "G": 0,
                  "O": 0
                }
                # TODO: Need to test these
                log_message = {
                    "mqtt_message": mqtt_message,
                    "three_d_point": asdict(three_d_point),
                }
                log_entry = {
                    "type": "published",
                    "message": log_message,
                    "timestamp": time()
                }
                logger.info(json.dumps(log_entry))
                iot_manager.publish(payload=json.dumps(mqtt_message))

        sleep(1 / 6)  # Wait for 1/6 seconds


if __name__ == "__main__":
    cwd = os.getcwd()

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

    subscribe_future = iot_manager.subscribe(topic=iot_manager.subscribe_topic, handler=on_message_received)
    subscribe_result = subscribe_future.result()
    print(f"Subscribed with {str(subscribe_result['qos'])}")

    if not received_all_event.is_set():
        print("Waiting to receive message.")

    # Start the periodic sending in a separate thread
    threading.Thread(target=send_detections_periodically, daemon=True).start()

    # Set a timeout for 6 hours (3 hours * 60 minutes/hour * 60 seconds/minute)

    # TODO: Ensure that this can run indefinitely and doesn't time out
    temp_received_count = 0
    while not received_all_event.is_set():
        if len(received_message) == 0:
            print("received_message is empty. Continuing...")
            sleep(0.5)
            continue

    received_all_event.wait()  # https://docs.python.org/3/library/threading.html#threading.Event.wait used with .set()

    disconnect_future = iot_manager.disconnect()
    disconnect_future.result()
    print("Disconnected!")
