import os

USE_OMAR_CERTS: bool = True

class MQTTMergerConfig:
    def __init__(self):

        self.camera_topic: str = "marvel_AUS/cameras"  # MQTT topic to listen where the cameras publish to
        self.device_topic: str = "marvel_AUS/pub"  # MQTT topic to publish to where the device listens to


        # Temp
        self.device_topic: str = "dalymount_IRL/pub"

        if os.name == 'nt':
            self.endpoint: str = "a3lkzcadhi1yzr-ats.iot.eu-west-1.amazonaws.com"
            # self.cert_path: str = "./certs/marvel-fov-test/certificate.pem.crt"
            # self.private_key_path: str = "./certs/marvel-fov-test/private.pem.key"
            # self.root_ca_path: str = "./certs/marvel-fov-test/AmazonRootCA1.pem"

            # Omars
            # TODO: temp test
            self.endpoint: str = "a13d7wu4wem7v1-ats.iot.eu-west-1.amazonaws.com"
            self.cert_path: str = "./certs/omar-certs-for-testing-old-firmware/certificate.pem.crt"
            self.private_key_path: str = "./certs/omar-certs-for-testing-old-firmware/private.pem.key"
            self.root_ca_path: str = "./certs/omar-certs-for-testing-old-firmware/root.pem"
        else:

            if not USE_OMAR_CERTS:
                self.endpoint: str = "a3lkzcadhi1yzr-ats.iot.ap-southeast-2.amazonaws.com"
                self.cert_path: str = "./certs/marvel-fov-test-sydney/certificate.pem.crt"
                self.private_key_path: str = "./certs/marvel-fov-test-sydney/private.pem.key"
                self.root_ca_path: str = "./certs/marvel-fov-test-sydney/AmazonRootCA1.pem"
            else:
                self.endpoint: str = "a13d7wu4wem7v1-ats.iot.ap-southeast-2.amazonaws.com"
                self.cert_path: str = "./certs/omar-aus/certificate.pem.crt"
                self.private_key_path: str = "./certs/omar-aus/private.pem.key"
                self.root_ca_path: str = "./certs/omar-aus/AmazonRootCA1.pem"
