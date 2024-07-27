"""
The send script for measruing latency between sending and receiving MQTT messages. To handle time synchronization,
we measure the round-trip latency by sending a message with a timestamp and receiving it back from the server.

Here's the setup:
The sender (on your laptop) sends a message with a unique ID to the "marvel_AUS/laptop_to_server" topic.
The listener (on SageMaker) receives this message and immediately sends it back to the "marvel_AUS/server_to_laptop" topic.
The sender receives the returned message and calculates the round-trip time.

The measured latency includes:
Time for the message to travel from your laptop to AWS IoT Core
Processing time in AWS IoT Core
Time for the message to travel from AWS IoT Core to the SageMaker instance
Processing time on the SageMaker instance
The return journey (SageMaker to AWS IoT Core to your laptop)
"""
import json
import time
import uuid
import threading
from aws_iot.IOTContext import IOTContext, IOTCredentials
from aws_iot.IOTClient import IOTClient
from config import MQTTMergerConfig


class MQTTLatencyMeasurer:
    def __init__(self):
        self.config = MQTTMergerConfig()
        self.send_topic = "marvel_AUS/ai_pub"
        self.receive_topic = "esp32/echo"
        self.log_file = "latency_log.txt"

        self.iot_credentials = IOTCredentials(
            cert_path=self.config.cert_path,
            client_id="iot_sender",
            endpoint=self.config.endpoint,
            region="ap-southeast-2",
            priv_key_path=self.config.private_key_path,
            ca_path=self.config.root_ca_path
        )
        self.iot_client = IOTClient(
            IOTContext(),
            self.iot_credentials,
            publish_topic=self.send_topic,
            subscribe_topic=self.receive_topic
        )
        self.latencies = []
        self.pending_messages = {}
        self.running = False
        self.connect_client()

    def connect_client(self):
        connect_future = self.iot_client.connect()
        connect_future.result()
        self.iot_client.subscribe(topic=self.receive_topic, handler=self.on_message_received)
        print("IOT client connected for sending and receiving!")

    def send_messages(self):
        while self.running:
            for _ in range(50):  # Send 20 messages
                message_id = str(uuid.uuid4())
                message = {"ID": message_id, "time": time.time()}
                self.pending_messages[message_id] = time.time()
                self.iot_client.publish(topic=self.send_topic, payload=json.dumps(message))
                time.sleep(0.05)  # 0.05 second delay between messages

            print("Sent 20 messages. Waiting for 30 seconds...")
            time.sleep(30)  # Wait for 30 seconds before the next batch

    def on_message_received(self, topic, payload, dup, qos, retain, **kwargs):
        receive_time = time.time()
        message = json.loads(payload)
        message_id = message['ID']
        if message_id in self.pending_messages:
            send_time = self.pending_messages[message_id]
            latency = (receive_time - send_time) * 1000  # Convert to milliseconds
            self.latencies.append(latency)
            log_entry = f"Message {message_id}: Latency = {latency:.2f} ms, Sent at {send_time}, Received at {receive_time}\n"
            self.log_to_file(log_entry)
            print(f"Round-trip latency for message {message_id}: {latency:.2f} ms")
            del self.pending_messages[message_id]

    def log_to_file(self, log_entry):
        with open(self.log_file, 'a') as f:
            f.write(log_entry)

    def disconnect_client(self):
        self.iot_client.disconnect()
        print("IOT client disconnected!")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.send_messages)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.disconnect_client()


def run_mqtt_sender():
    sender = MQTTLatencyMeasurer()
    try:
        sender.start()
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping MQTT Sender...")
    finally:
        sender.stop()


if __name__ == "__main__":
    run_mqtt_sender()