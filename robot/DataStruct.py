from dataclasses import dataclass
import json


@dataclass
class MultimodalObject:
    def __init__(self,
                 uid: str = None,
                 timestamp: str = "2025-01-01T00:00:00Z",
                 olf_device_uid: str = "XXXXXX",
                 aux_device_uid: str = "XXXXXX",
                 img_path_list: list = [],
                 img_uid_list: list = [],
                 inertial_data: dict = {},
                 battery_level: float = 100.0,
                 battery_voltage: float = 0.0,
                 rh: float = 0.0,
                 temp_c: float = 0.0,
                 pressure_hpa: float = 0.0,
                 light_lux: float = 0.0,
                 audio_db: float = 0.0,
                 opu_script: str = "",
                 co_ppb: float = 0.0,
                 co2_ppb: float = 0.0,
                 ch4_ppb: float = 0.0,
                 c2h5oh_ppb: float = 0.0,
                 c3h8_ppb: float = 0.0,
                 c4h10_ppb: float = 0.0,
                 h2_ppb: float = 0.0,
                 h2o_ppb: float = 0.0,
                 h2s_ppb: float = 0.0,
                 no_ppb: float = 0.0,
                 no2_ppb: float = 0.0,
                 nh3_ppb: float = 0.0,
                 nh4_ppb: float = 0.0,
                 o3_ppb: float = 0.0,
                 voc_ppb: float = 0.0,
                 loc_lat: float = 0.0,
                 loc_lon: float = 0.0,
                 loc_alt: float = 0.0
                 ):
        """
        A multimodal object that can be used to represent and fuse different types of data.
        Olfactory data, vision data, inertial data, lumosity data, audio data, and environmental data are all captured.

        :param uid: Unique identifier for the object.
        :param timestamp: Timestamp of the data.
        :param olf_device_uid: Unique identifier for the olfactory device.
        :param aux_device_uid: Unique identifier for the auxiliary device.
        :param img_path_list: List of image paths.
        :param img_uid_list: List of image unique identifiers.
        :param inertial_data: Dictionary containing 9-DoF inertial data (x, y, z, pitch, yaw, roll, ax, ay, az, gx, gy, gz).
        :param battery_level: Battery level in percentage.
        :param battery_voltage: Battery voltage in volts.
        :param rh: Relative humidity in percentage.
        :param temp_c: Temperature in degrees Celsius.
        :param pressure_hpa: Pressure in hectopascals.
        :param light_lux: Light intensity in lux.
        :param audio_db: Audio level in decibels.
        :param opu_script: Optional script associated with chemical processor, if executed.
        :param co_ppb: Carbon monoxide concentration in parts per billion.
        :param co2_ppb: Carbon dioxide concentration in parts per billion.
        :param ch4_ppb: Methane concentration in parts per billion.
        :param c2h5oh_ppb: Ethanol concentration in parts per billion.
        :param c3h8_ppb: Propane concentration in parts per billion.
        :param c4h10_ppb: Butane concentration in parts per billion.
        :param h2_ppb: Hydrogen concentration in parts per billion.
        :param h2o_ppb: Water vapor concentration in parts per billion.
        :param h2s_ppb: Hydrogen sulfide concentration in parts per billion.
        :param no_ppb: Nitric oxide concentration in parts per billion.
        :param no2_ppb: Nitrogen dioxide concentration in parts per billion.
        :param nh3_ppb: Ammonia concentration in parts per billion.
        :param nh4_ppb: Ammonium concentration in parts per billion.
        :param o3_ppb: Ozone concentration in parts per billion.
        :param voc_ppb: Volatile organic compounds concentration in parts per billion.
        :param loc_lat: GPS latitude of the location.
        :param loc_lon: GPS longitude of the location.
        :param loc_alt: GPS altitude of the location.
        """
        self.uid = uid
        self.timestamp = timestamp
        self.olf_device_uid = olf_device_uid
        self.aux_device_uid = aux_device_uid
        self.img_path_list = img_path_list
        self.img_uid_list = img_uid_list
        self.inertial_data = inertial_data
        self.battery_level = battery_level
        self.battery_voltage = battery_voltage
        self.rh = rh
        self.temp_c = temp_c
        self.pressure_hpa = pressure_hpa
        self.light_lux = light_lux
        self.audio_db = audio_db
        self.opu_script = opu_script
        self.co_ppb = co_ppb
        self.co2_ppb = co2_ppb
        self.ch4_ppb = ch4_ppb
        self.c2h5oh_ppb = c2h5oh_ppb
        self.c3h8_ppb = c3h8_ppb
        self.c4h10_ppb = c4h10_ppb
        self.h2_ppb = h2_ppb
        self.h2o_ppb = h2o_ppb
        self.h2s_ppb = h2s_ppb
        self.no_ppb = no_ppb
        self.no2_ppb = no2_ppb
        self.nh3_ppb = nh3_ppb
        self.nh4_ppb = nh4_ppb
        self.o3_ppb = o3_ppb
        self.voc_ppb = voc_ppb
        self.loc_lat = loc_lat
        self.loc_lon = loc_lon
        self.loc_alt = loc_alt

    def __del__(self):
        print(f'\n\nMultimodalObject with UID {self.uid} deleted')
        del self

    def upload_object(self):
        """
        Upload the multimodal object to the server.
        This method should be implemented to send the object data to a remote server.
        """
        pass

    def download_object():
        """
        Download the multimodal object from the server.
        This method should be implemented to fetch the object data from a remote server.
        """
        pass

    def to_json(self) -> str:
        """
        Convert the multimodal object to a JSON string.
        :return: JSON string representation of the object.
        """
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(json_str: str) -> 'MultimodalObject':
        """
        Create a multimodal object from a JSON string.
        :param json_str: JSON string representation of the object.
        :return: An instance of MultimodalObject.
        """
        data = json.loads(json_str)
        return MultimodalObject(**data)

    def print_object(self):
        """
        Print the multimodal object data properties - may be helpful for debugging.
        """
        print(f"Multimodal Object UID: {self.uid}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Olfactory Device UID: {self.olf_device_uid}")
        print(f"Auxiliary Device UID: {self.aux_device_uid}")
        print(f"Image Paths: {self.img_path_list}")
        print(f"Image UIDs: {self.img_uid_list}")
        print(f"Inertial Data: {self.inertial_data}")
        print(f"Relative Humidity: {self.rh}%")
        print(f"Temperature: {self.temp_c}°C")
        print(f"Pressure: {self.pressure_hpa} hPa")
        print(f"Light Intensity: {self.light_lux} lux")
        print(f"Audio Level: {self.audio_db} dB")
        print(f"CO Concentration: {self.co_ppb} ppb")
        print(f"CO2 Concentration: {self.co2_ppb} ppb")
        print(f"Methane Concentration: {self.ch4_ppb} ppb")
        print(f"Ethanol Concentration: {self.c2h5oh_ppb} ppb")
        print(f"Propane Concentration: {self.c3h8_ppb} ppb")
        print(f"Butane Concentration: {self.c4h10_ppb} ppb")
        print(f"Hydrogen Concentration: {self.h2_ppb} ppb")
        print(f"Water Vapor Concentration: {self.h2o_ppb} ppb")
        print(f"Hydrogen Sulfide Concentration: {self.h2s_ppb} ppb")
        print(f"Nitric Oxide Concentration: {self.no_ppb} ppb")
        print(f"Nitrogen Dioxide Concentration: {self.no2_ppb} ppb")
        print(f"Ammonia Concentration: {self.nh3_ppb} ppb")
        print(f"Ammonium Concentration: {self.nh4_ppb} ppb")
        print(f"Ozone Concentration: {self.o3_ppb} ppb")
        print(f"VOC Concentration: {self.voc_ppb} ppb")
        print(f"Location - Latitude: {self.loc_lat}, Longitude: {self.loc_lon}, Altitude: {self.loc_alt}")

        # Return self for method chaining if needed
        return self