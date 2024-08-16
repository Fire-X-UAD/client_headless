import argparse
import serial
import serial.tools.list_ports
import socket
import threading
import time

class HeadlessClient:
    def __init__(self, vision_ip="127.0.0.1", vision_port=5005, basestation_ip=None, basestation_port=None, send_interval=0.1, standalone_mode=False):
        """
        Initializes a new instance of the HeadlessClient class.

        Args:
            vision_ip (str, optional): The IP address for Vision UDP.
            vision_port (int, optional): The port for Vision UDP.
            basestation_ip (str, optional): The IP address for Basestation TCP. Defaults to None.
            basestation_port (int, optional): The port for Basestation TCP. Defaults to None.
            send_interval (float, optional): The interval for sending data in seconds. Defaults to 0.1.
            standalone_mode (bool, optional): Enable standalone mode. Defaults to False.

        Returns:
            None
        """
        self.arduino_baud = 9600  # Fixed baud rate
        self.vision_ip = vision_ip
        self.vision_port = vision_port
        self.basestation_ip = basestation_ip
        self.basestation_port = basestation_port
        self.send_interval = send_interval
        self.standalone_mode = standalone_mode
        self.arduino_port = None
        self.vision_socket = None
        self.basestation_client = None
        self.data_arduino = ""
        self.data_vision = ""
        self.data_basestation = ""
        self.is_arduino_ever_used = False
        self.is_vision_ever_used = False
        self.is_basestation_ever_used = False

    def search_arduino_port(self):
        """
        Search for the Arduino port.

        This function iterates over all the available serial ports and checks if the description of each port contains the word "Arduino". If a matching port is found, its device name is returned. If no matching port is found, None is returned.

        Returns:
            str or None: The device name of the Arduino port if found, None otherwise.
        """
        # Search for Arduino port
        for port in serial.tools.list_ports.comports():
            if "Arduino" in port.manufacturer or "Microsoft" in port.manufacturer:
                return port.device
        return None

    def connect_arduino(self):
        """
        Connects to an Arduino device.

        This function continuously tries to connect to an Arduino device by searching for the Arduino port.
        It uses the `search_arduino_port` method to find the Arduino port name. If the port is found, it creates a serial connection
        using the `serial.Serial` class with the specified baud rate and timeout. If the connection is successful, it sets the `is_arduino_ever_used`
        flag to True and prints a message indicating the successful connection. It then calls the `read_arduino_data` method to start reading data from the Arduino.

        If the Arduino port is not found or there is an exception during the connection, it prints an error message and waits for 5 seconds before trying again.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        print("Connecting to Arduino...")
        while True:
            try:
                arduino_port_name = self.search_arduino_port()
                if not arduino_port_name:
                    raise Exception("Arduino port not found")
                self.arduino_port = serial.Serial(
                    port=arduino_port_name,
                    baudrate=self.arduino_baud,
                    timeout=1
                )
                self.is_arduino_ever_used = True
                print("Connected to Arduino")
                self.read_arduino_data()
            except Exception as e:
                print(f"Error connecting to Arduino: {e}")
                time.sleep(5)  # Wait before retrying

    def read_arduino_data(self):
        """
        Continuously reads data from the Arduino port.

        This function reads data from the Arduino port in a loop until the port is closed or an error occurs.
        It uses the `arduino_port` attribute of the class to establish a serial connection with the Arduino.
        The function reads a line of data from the Arduino using the `readline()` method and decodes it from bytes to a string.
        The resulting string is then stripped of leading and trailing whitespace.
        If the data is successfully read, it is printed to the console with a prefix "Arduino Data:".
        If an error occurs while reading from the Arduino, the function catches the exception, prints an error message to the console,
        sets the `arduino_port` attribute to None to close the port, and breaks out of the loop.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        while self.arduino_port and self.arduino_port.is_open:
            try:
                self.data_arduino = self.arduino_port.readline().decode().strip()
                print(f"Arduino Data: {self.data_arduino}")
            except Exception as e:
                print(f"Error reading from Arduino: {e}")
                self.arduino_port = None
                break

    def connect_vision(self):
        """
        Continuously connects to the Vision server and reads data from it.

        This function establishes a UDP socket connection to the Vision server and binds it to the specified IP address and port.
        It sets the `is_vision_ever_used` flag to True to indicate that the Vision server has been used.
        It prints a message indicating that the connection to the Vision server has been established.
        It then calls the `read_vision_data` method to continuously read data from the Vision server.

        If an exception occurs while connecting to the Vision server, it prints an error message to the console and waits for 5 seconds before retrying.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        print("Connecting to Vision...")
        while True:
            try:
                self.vision_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.vision_socket.bind((self.vision_ip, self.vision_port))
                self.is_vision_ever_used = True
                print("Connected to Vision")
                self.read_vision_data()
            except Exception as e:
                print(f"Error connecting to Vision: {e}")
                time.sleep(5)  # Wait before retrying

    def read_vision_data(self):
        """
        Reads data from the vision socket and updates the vision data attribute.

        This function continuously reads data from the vision socket until the socket is closed or an error occurs.
        It uses the `vision_socket` attribute of the class to establish a socket connection with the vision server.
        The function receives data from the vision server using the `recvfrom()` method and decodes it from bytes to a string.
        The resulting string is then stripped of leading and trailing whitespace.
        If the data is successfully read, it is printed to the console with a prefix "Vision Data:".
        If an error occurs while reading from the vision socket, the function catches the exception, prints an error message to the console,
        sets the `vision_socket` attribute to None to close the socket, and breaks out of the loop.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        while self.vision_socket:
            try:
                data, _ = self.vision_socket.recvfrom(1024)
                self.data_vision = data.decode().strip()
                print(f"Vision Data: {self.data_vision}")
            except Exception as e:
                print(f"Error reading from Vision: {e}")
                self.vision_socket = None
                break

    def connect_basestation(self):
        """
        Continuously tries to connect to the Basestation and reads data from it.

        This function establishes a TCP connection with the Basestation by creating a socket and calling the `connect()` method.
        It sets the `is_basestation_ever_used` flag to True if the connection is successful.
        If the connection fails, it prints an error message and waits for 5 seconds before retrying.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        print("Connecting to Basestation...")

        while True:
            try:
                self.basestation_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.basestation_client.connect((self.basestation_ip, self.basestation_port))
                self.is_basestation_ever_used = True
                print("Connected to Basestation")
                self.read_basestation_data()
            except Exception as e:
                print(f"Error connecting to Basestation: {e}")
                time.sleep(5)  # Wait before retrying

    def read_basestation_data(self):
        """
        Continuously reads data from the Basestation until the connection is closed.

        This function reads data from the Basestation by calling the `recv()` method on the `basestation_client` socket.
        The received data is decoded and split at the '#' character. The first part of the split data is assigned to the `data_basestation` attribute.
        The received data is then printed to the console.

        If an exception occurs during the reading process, the error message is printed to the console and the `basestation_client` socket is set to None.
        The loop is then broken and the function returns.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        while self.basestation_client and self.basestation_client.fileno() != -1:
            try:
                data = self.basestation_client.recv(1024).decode()
                self.data_basestation = data.split('#')[0]
                print(f"Basestation Data: {self.data_basestation}")
            except Exception as e:
                print(f"Error reading from Basestation: {e}")
                self.basestation_client = None
                break

    def send_data(self):
        """
        Sends data to the Basestation and Arduino indefinitely.

        This function continuously sends data to the Basestation and Arduino. It first checks if the Basestation client is connected and not closed. If it is, it checks if there is data to send to the Basestation. If there is, it sends the data to the Basestation. If an exception occurs during the sending process, the error message is printed to the console.

        Next, it checks if the Arduino port is open. If it is, it constructs the vision data by checking if the vision socket is available. If the vision socket is available, it uses the vision data. Otherwise, it uses "-1" as the vision data. It then sends the constructed data to the Arduino. If an exception occurs during the sending process, the error message is printed to the console.

        If the standalone mode is enabled and the Arduino port is open, it constructs the vision data in the same way as before. It then sends the constructed data to the Arduino, along with additional "-1" values to indicate standalone mode. If an exception occurs during the sending process, the error message is printed to the console.

        Finally, it pauses for the specified send interval before repeating the process.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        while True:
            try:
                if self.basestation_client and self.basestation_client.fileno() != -1:
                    if self.data_arduino:
                        self.basestation_client.sendall(f"{self.data_arduino}#".encode())
            except Exception as e:
                print(f"Error sending to Basestation: {e}")

            try:
                if self.arduino_port and self.arduino_port.is_open:
                    vision_data = self.data_vision if self.vision_socket else "-1"
                    self.arduino_port.write(f"{self.data_basestation};{vision_data}#".encode())
            except Exception as e:
                print(f"Error sending to Arduino: {e}")

            try:
                if self.standalone_mode and self.arduino_port and self.arduino_port.is_open:
                    vision_data = self.data_vision if self.vision_socket else "-1"
                    self.arduino_port.write(f"1;-1;-1;-1;-1;-1;-1;{vision_data}#".encode())
            except Exception as e:
                print(f"Error sending standalone data to Arduino: {e}")

            time.sleep(self.send_interval)

    def start(self):
        """
        Starts the execution of the program by creating and starting multiple threads.

        This function starts the execution of the program by creating and starting multiple threads. It creates and starts a thread for connecting to the Arduino, a thread for connecting to the vision, and a thread for sending data. If the program is not running in standalone mode, it also creates and starts a thread for connecting to the basestation.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        threading.Thread(target=self.connect_arduino, daemon=True).start()
        threading.Thread(target=self.connect_vision, daemon=True).start()
        if not self.standalone_mode:
            threading.Thread(target=self.connect_basestation, daemon=True).start()
        else:
            print("Standalone mode is enabled. Skipping basestation connection.")
        threading.Thread(target=self.send_data, daemon=True).start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Client for FireX")
    parser.add_argument("--vision-ip", "-i", type=str, default="127.0.0.1", help="IP address for Vision UDP")
    parser.add_argument("--vision-port", "-t", type=int, default=5005, help="Port for Vision UDP")
    parser.add_argument("--basestation-ip", "-b", type=str, help="IP address for Basestation TCP")
    parser.add_argument("--basestation-port", "-p", type=int, help="Port for Basestation TCP")
    parser.add_argument("--send-interval", type=float, default=0.1, help="Interval for sending data in seconds (default: 0.1s)")
    parser.add_argument("--standalone-mode", "-s", action='store_true', help="Enable standalone mode")

    # add alias for standalone mode

    args = parser.parse_args()

    if not args.standalone_mode and (not args.basestation_ip or not args.basestation_port):
        parser.error("--basestation-ip and --basestation-port are required unless --standalone-mode is set")

    client = HeadlessClient(
        vision_ip=args.vision_ip,
        vision_port=args.vision_port,
        basestation_ip=args.basestation_ip,
        basestation_port=args.basestation_port,
        send_interval=args.send_interval,
        standalone_mode=args.standalone_mode
    )
    client.start()

    # Keep the main thread alive to allow the daemon threads to run
    while True:
        time.sleep(1)
