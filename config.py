import os


class MQTTMergerConfig:
    def __init__(self):

        self.camera_topic: str = "marvel_AUS/cameras"  # MQTT topic to listen where the cameras publish to
        self.device_topic: str = "marvel_AUS/pub"  # MQTT topic to publish to where the device listens to

        if os.name == 'nt':
            self.camera_topic = "cameras/bohs"
            self.endpoint: str = "a3lkzcadhi1yzr-ats.iot.eu-west-1.amazonaws.com"
            self.cert_path: str = "./certs/marvel-fov-test/certificate.pem.crt"
            self.private_key_path: str = "./certs/marvel-fov-test/private.pem.key"
            self.root_ca_path: str = "./certs/marvel-fov-test/AmazonRootCA1.pem"
        else:
            self.endpoint: str = "a3lkzcadhi1yzr-ats.iot.ap-southeast-2.amazonaws.com"
            self.cert_path: str = "./certs/marvel-fov-test-sydney/certificate.pem.crt"
            self.private_key_path: str = "./certs/marvel-fov-test-sydney/private.pem.key"
            self.root_ca_path: str = "./certs/marvel-fov-test-sydney/AmazonRootCA1.pem"

