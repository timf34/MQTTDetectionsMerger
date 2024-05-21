import json
import os
import threading
from time import time, sleep
import logging
from logging import Formatter, FileHandler
from typing import Union, Dict
from aws_iot.IOTClient import IOTClient
from aws_iot.IOTContext import IOTContext, IOTCredentials
from config import MQTTMergerConfig

NUM_MESSAGES = 1000000

received_count = 0
elapsed_time = 0
received_message: str = ""
received_all_event = threading.Event()

config = MQTTMergerConfig()

# Start a timer at 0 seconds
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
    print(f"Received message from topic '{topic}' at {time()}: {payload}")

    global received_count
    global received_message
    global elapsed_time

    received_count += 1
    received_message = payload

    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Received {received_count} messages in {elapsed_time} seconds")

    # Log the received message and timestamp
    log_entry = {
        "topic": topic,
        "message": payload.decode('utf-8') if isinstance(payload, bytes) else payload,
        "timestamp": elapsed_time
    }
    logger.info(json.dumps(log_entry))

    if received_count == NUM_MESSAGES:
        received_all_event.set()

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

    # Set a timeout for 3 hours (3 hours * 60 minutes/hour * 60 seconds/minute)
    timeout = 3 * 60 * 60
    end_time = start_time + timeout

    temp_received_count = 0
    while time() < end_time and not received_all_event.is_set():
        # If received message is of type bytes, decode it.
        if isinstance(received_message, bytes):
            if received_message != '':  # Check if empty... the first one probs will be. Note: '' works, "" doesn't.
                received_message = received_message.decode("utf-8")
        elif len(received_message) == 0:
            print("received_message is empty. Continuing...")
            continue

        received_message_json = json.loads(received_message)
        received_message_json["timestamp"] = elapsed_time  # This might be better off with time.time(), or whatever would match the message sent!
        print("received_message_json post whatever: ", received_message_json, "\n")

        # Update the detections dict
        # update_detections_dict(received_message_json)

        # iot_manager.publish(topic=iot_manager.publish_topic, payload=json.dumps(detections))  # Note: detections will be the triangulated coords
        temp_received_count = received_count

        # Wait until a new message is received.
        while temp_received_count == received_count and time() < end_time:
            sleep(0.01)

    print("outside of while loop")
    received_all_event.wait()  # https://docs.python.org/3/library/threading.html#threading.Event.wait used with .set()

    disconnect_future = iot_manager.disconnect()
    disconnect_future.result()
    print("Disconnected!")
