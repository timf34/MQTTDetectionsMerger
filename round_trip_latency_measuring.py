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
        self.reconnect_interval = 3600  # 1 hour

        self.iot_credentials = IOTCredentials(
            cert_path=self.config.cert_path,
            client_id="iot_sender",
            endpoint=self.config.endpoint,
            region="ap-southeast-2",
            priv_key_path=self.config.private_key_path,
            ca_path=self.config.root_ca_path
        )
        self.iot_client = None
        self.pending_messages = {}
        self.running = False
        self.connect_client()

    def connect_client(self):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.iot_client = IOTClient(
                    IOTContext(),
                    self.iot_credentials,
                    publish_topic=self.send_topic,
                    subscribe_topic=self.receive_topic
                )
                connect_future = self.iot_client.connect()
                connect_future.result()
                self.iot_client.subscribe(topic=self.receive_topic, handler=self.on_message_received)
                print("IOT client connected for sending and receiving!")
                return
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait before retrying
                else:
                    raise

    def send_messages(self):
        last_reconnect_time = time.time()

        while self.running:
            current_time = time.time()

            # Periodic reconnection
            if current_time - last_reconnect_time > self.reconnect_interval:
                self.reconnect()
                last_reconnect_time = current_time

            for _ in range(50):  # Send 50 messages
                message_id = str(uuid.uuid4())
                message = {"ID": message_id, "time": time.time()}
                self.pending_messages[message_id] = time.time()
                try:
                    self.iot_client.publish(topic=self.send_topic, payload=json.dumps(message))
                except Exception as e:
                    print(f"Failed to publish message: {str(e)}")
                time.sleep(0.05)  # 0.05 second delay between messages

            print("Sent 50 messages. Waiting for 30 seconds...")
            time.sleep(30)  # Wait for 30 seconds before the next batch

    def on_message_received(self, topic, payload, dup, qos, retain, **kwargs):
        receive_time = time.time()
        try:
            message = json.loads(payload)
            message_id = message['ID']
            if message_id in self.pending_messages:
                send_time = self.pending_messages[message_id]
                latency = (receive_time - send_time) * 1000  # Convert to milliseconds
                log_entry = f"Message {message_id}: Latency = {latency:.2f} ms, Sent at {send_time}, Received at {receive_time}\n"
                self.log_to_file(log_entry)
                print(f"Round-trip latency for message {message_id}: {latency:.2f} ms")
                del self.pending_messages[message_id]
        except Exception as e:
            print(f"Error processing received message: {str(e)}")

    def log_to_file(self, log_entry):
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")

    def reconnect(self):
        print("Attempting to reconnect...")
        try:
            self.disconnect_client()
        except:
            pass
        self.connect_client()

    def disconnect_client(self):
        if self.iot_client:
            self.iot_client.disconnect()
            print("IOT client disconnected!")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.send_messages)
        self.thread.start()
        self.cleanup_thread = threading.Thread(target=self.cleanup_pending_messages)
        self.cleanup_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        if hasattr(self, 'cleanup_thread'):
            self.cleanup_thread.join()
        self.disconnect_client()

    def cleanup_pending_messages(self):
        while self.running:
            current_time = time.time()
            to_remove = [msg_id for msg_id, send_time in self.pending_messages.items() if current_time - send_time > 60]
            for msg_id in to_remove:
                del self.pending_messages[msg_id]
            time.sleep(80)  # Check every 80 seconds


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
