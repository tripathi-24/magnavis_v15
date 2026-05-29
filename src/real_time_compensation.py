import sys
import os
import serial
import threading
import time
import re
from datetime import datetime, timezone
import requests
import collections # Added for deque
import mysql.connector # Added for database
from uuid import getnode as get_mac
import subprocess

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QHBoxLayout, QLabel, QLineEdit,
                             QComboBox, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer

import pyqtgraph as pg

# --- Global Configurations ---
WEB_API_URL = "http://127.0.0.1:8000/quantum/magnavis/api/magnetic-data/"
API_POST_INTERVAL_MS = 1000 # 1 second in milliseconds

DB_CONFIG = {
    'host': '208.109.66.54',
    'user': 'magnavis',
    'password': 'magnavis@iitk123',
    'database': 'kts-webdb-dev',
    'port': 3306
}
MAGNETIC_DATA_TABLE = "KTSWebsite_magneticdatamodel" # Confirmed table name

# --- Serial Reader Thread ---
class SerialReader(QThread):
    data_received = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, port, baud_rate):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.running = False
        self.serial_connection = None

    def run(self):
        self.running = True
        try:
            self.serial_connection = serial.Serial(self.port, self.baud_rate, timeout=1)
            print(f"Opened serial port: {self.port} at {self.baud_rate} baud")
            while self.running:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    if line:
                        try:
                            values_str = line.split(',')
                            # print('values_str', values_str)
                            if len(values_str) == 9:
                                try:
                                    bx = float(values_str[0])
                                    by = float(values_str[1])
                                    bz = float(values_str[2])
                                    lat = float(values_str[3])
                                    lon = float(values_str[4])
                                    alt = float(values_str[5])
                                    thetax = float(values_str[6])
                                    thetay = float(values_str[7])
                                    thetaz = float(values_str[8])
                                except:
                                    continue
                                self.data_received.emit([bx, by, bz, lat, lon, alt, thetax, thetay, thetaz])
                            else:
                                # put some retries logic before straightaway throwing error
                                self.error_occurred.emit(f"Malformed data: {line}")
                        except ValueError:
                            self.error_occurred.emit(f"Could not parse data: {line}")
                time.sleep(0.001)
        except serial.SerialException as e:
            self.error_occurred.emit(f"Serial port error: {e}")
            print(f"Serial port error in SerialReader: {e}")
        except Exception as e:
            self.error_occurred.emit(f"Unexpected error in SerialReader: {e}")
            print(f"Unexpected error in SerialReader: {e}")
        finally:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
                print("Serial port closed.")
            self.running = False

    def stop(self):
        self.running = False
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.wait()

# --- Database Writer Thread (Per-Insert Connection Strategy) ---
import collections
import mysql.connector
import time # Ensure time is imported

class DatabaseWriter(QThread):
    db_status_signal = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.data_queue = collections.deque()
        self.batch_size = 200 # Define a batch size, or process all available.
                              # For "last inserted till now," we'll process all available.

    def run(self):
        self.running = True
        self.db_status_signal.emit("Database writer started (batch connections).")
        print("DatabaseWriter thread: RUN method entered (batch connections).")

        try:
            while self.running:
                if self.data_queue: # Only proceed if there's data to process
                    # Collect all available data from the queue into a list
                    data_points_to_insert = []
                    while self.data_queue: # Pop all items currently in queue
                        data_points_to_insert.append(self.data_queue.popleft())

                    if data_points_to_insert: # Ensure the list is not empty
                        try:
                            # print(f"DatabaseWriter thread: Processing batch of {len(data_points_to_insert)} points for insert.")
                            # print("from", data_points_to_insert[0], "to", data_points_to_insert[-1])
                            self._insert_data(data_points_to_insert) # Now sends a list
                            self.db_status_signal.emit(f"Batch of {len(data_points_to_insert)} points inserted.")
                        except Exception as e: # Catch any errors during insert, including connection issues
                            self.error_occurred.emit(f"DB Batch Insert/Connection Error: {e}")
                            print(f"DatabaseWriter thread: CAUGHT ERROR during batch insert: {e}")
                else:
                    time.sleep(0.1) # Small sleep to prevent busy-waiting if queue is empty

        except Exception as e:
            self.error_occurred.emit(f"Critical error in DatabaseWriter thread: {e}. Stopping.")
            print(f"DatabaseWriter thread: CRITICAL ERROR in run method: {e}")
        finally:
            self.db_status_signal.emit("Database writer stopped.")
            self.running = False
            print("DatabaseWriter thread: RUN method exiting.")

    def _insert_data(self, data_points_list): # Now accepts a list of dictionaries
        mydb = None
        mycursor = None
        try:
            print(f"Attempting to connect for batch insert ({len(data_points_list)} points): {DB_CONFIG['host']}:{DB_CONFIG['port']}")
            time.sleep(0.01) # Small delay for stability

            mydb = mysql.connector.connect(
                host=DB_CONFIG['host'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database'],
                port=int(DB_CONFIG['port']),
                connection_timeout=5
            )
            mycursor = mydb.cursor()
            print("Connection for batch insert successful.")

            sql = f"""
            INSERT INTO {MAGNETIC_DATA_TABLE}
            ( b_x, b_y, b_z, timestamp, lat, lon, alt, theta_x, theta_y, theta_z, sensor_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            # Prepare the list of tuples for executemany
            values_to_insert = []
            for dp in data_points_list:
                values_to_insert.append((
                    dp["b_x"],
                    dp["b_y"],
                    dp["b_z"],
                    dp["timestamp"],
                    dp['lat'],
                    dp['lon'],
                    dp['alt'],
                    dp['thetax'],
                    dp['thetay'],
                    dp['thetaz'],
                    dp["sensor_id"]
                ))
            
            # print('sql, values_to_insert', sql, values_to_insert)
            mycursor.executemany(sql, values_to_insert) # Use executemany
            mydb.commit()
            print(f"Batch of {len(data_points_list)} points inserted successfully.")

        except mysql.connector.Error as err:
            raise Exception(f"MySQL Error during batch insert: {err}")
        except Exception as e:
            raise Exception(f"General Error during batch insert: {e}")
        finally:
            if mycursor:
                mycursor.close()
                print("Cursor closed after batch insert.")
            if mydb and mydb.is_connected():
                mydb.close()
                print("Connection closed after batch insert.")

    def _close_db_connection(self):
        # This remains largely unused for the per-insert connection strategy
        print("DatabaseWriter: _close_db_connection called (no active persistent connection).")
        pass

    def add_data_to_queue(self, data_point):
        """Called from the main thread to add a single data point to the queue."""
        self.data_queue.append(data_point)

    def stop(self):
        self.running = False
        self.wait()

# --- Main Window Class ---
class MainWindow(QMainWindow):
    # api_post_result = pyqtSignal(bool, str)
    # send_data_to_db_writer = pyqtSignal(dict) # New signal to send data to DB writer

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Serial Data Plotter, Logger & API Sender")
        self.setGeometry(100, 100, 1000, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.serial_reader = None
        self.log_file = None
        self.api_timer = QTimer(self)
        self.db_writer_thread = None # Added for DatabaseWriter instance

        self.init_ui()
        self.init_plot()
        self.data_buffer = {'current_time_pc': [], 'b_x': [], 'b_y': [], 'b_z': [], 'lat': [], 'lon': [], 'alt': [], 'thetax': [], 'thetay': [], 'thetaz': [], 'sensor_id': []}

        self.max_points = 500
        self.latest_api_data = None

        # self.api_post_result.connect(self.handle_api_post_result)

    def init_ui(self):
        serial_config_layout = QHBoxLayout()

        serial_config_layout.addWidget(QLabel("Port:"))
        self.port_input = QComboBox()
        self.populate_ports()
        serial_config_layout.addWidget(self.port_input)

        serial_config_layout.addWidget(QLabel("Baud Rate:"))
        self.baud_rate_input = QLineEdit("115200")
        serial_config_layout.addWidget(self.baud_rate_input)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_serial)
        serial_config_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_serial)
        self.stop_button.setEnabled(False)
        serial_config_layout.addWidget(self.stop_button)

        self.reset_plot_button = QPushButton("Reset Plot View")
        self.reset_plot_button.clicked.connect(self.reset_plot_view)
        serial_config_layout.addWidget(self.reset_plot_button)

        self.main_layout.addLayout(serial_config_layout)

        log_config_layout = QHBoxLayout()
        log_config_layout.addWidget(QLabel("Log File Prefix:"))
        self.log_file_prefix_input = QLineEdit("sensor_data")
        log_config_layout.addWidget(self.log_file_prefix_input)
        self.main_layout.addLayout(log_config_layout)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setWordWrap(True)
        self.main_layout.addWidget(self.status_label)

    def populate_ports(self):
        ports = []
        if sys.platform.startswith('win'):
            for i in range(256):
                try:
                    s = serial.Serial(f'COM{i}')
                    s.close()
                    ports.append(f'COM{i}')
                except (serial.SerialException, OSError):
                    pass
        elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
            import glob
            usb_ports = glob.glob('/dev/ttyUSB*')
            acm_ports = glob.glob('/dev/ttyACM*')
            bluetooth_ports = glob.glob('/dev/tty.Bluetooth-*')
            ports = sorted(usb_ports + acm_ports + bluetooth_ports)
        self.port_input.addItems(ports)

    def init_plot(self):
        self.plot_widget = pg.PlotWidget()
        self.main_layout.addWidget(self.plot_widget)
        self.plot_widget.setTitle("Real-time Sensor Data")
        self.plot_widget.setLabel('bottom', 'Time (t)')
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.addLegend()

        self.curve_x = self.plot_widget.plot(pen='r', name='x vs t')
        self.curve_y = self.plot_widget.plot(pen='g', name='y vs t')
        self.curve_z = self.plot_widget.plot(pen='b', name='z vs t')

    def start_serial(self):
        port = self.port_input.currentText()
        try:
            baud_rate = int(self.baud_rate_input.text())
        except ValueError:
            self.status_label.setText("Status: Invalid Baud Rate!")
            return

        if not port:
            self.status_label.setText("Status: Please select a serial port.")
            return

        self.latest_api_data = None
        self.curve_x.clear()
        self.curve_y.clear()
        self.curve_z.clear()

        self.serial_reader = SerialReader(port, baud_rate)
        self.serial_reader.data_received.connect(self.update_data)
        self.serial_reader.error_occurred.connect(self.handle_error)
        self.counter = time.time()
        self.serial_reader.start()
        #time.sleep(0.1)

        # # --- DATABASE WRITER THREAD START ---
        # self.db_writer_thread = DatabaseWriter()
        # self.db_writer_thread.db_status_signal.connect(lambda msg: self.status_label.setText(f"DB Status: {msg}"))
        # self.db_writer_thread.error_occurred.connect(self.handle_error)
        # self.send_data_to_db_writer.connect(self.db_writer_thread.add_data_to_queue)
        # self.db_writer_thread.start()
        # --- END DATABASE WRITER THREAD START ---

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.log_file_prefix_input.text()}_{timestamp}.csv"
            self.log_file_start_time = timestamp
            self.log_file = open(filename, 'w')
            # timestamp, bx, by, bz, lat, lon, altitude, thetax, thetay, thetaz, sensor_id
            self.log_file.write("timestamp_pc,b_x,b_y,b_z,lat,lon,alt,thetax,thetay,thetaz,sensor_id\n")
            print(f"Logging data to: {filename}")
        except IOError as e:
            self.status_label.setText(f"Status: Could not open log file: {e}")
            self.stop_serial()
            return

        # self.api_timer.timeout.connect(self.send_data_to_api)
        # self.api_timer.start(API_POST_INTERVAL_MS)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(f"Status: Reading from {port} at {baud_rate} baud, sending to API & DB...")

    def stop_serial(self):
        if self.serial_reader:
            self.serial_reader.stop()
            self.serial_reader = None

        if self.db_writer_thread: # Stop DB writer thread
            self.db_writer_thread.stop()
            self.db_writer_thread = None

        if self.log_file:
            self.log_file.close()
            self.log_file = None
            print("Log file closed.")

        self.api_timer.stop()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped.")

    def update_data(self, data):
        # print('data=', data)
        bx_val, by_val, bz_val, lat_val, lon_val, alt_val, thetax_val, thetay_val, thetaz_val = data
        current_time_pc = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        sensor_id = 'S' + str(get_mac()) + '_' + self.log_file_start_time

        # --- API Payload ---
        self.latest_api_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(), # UTC timestamp
            "b_x": bx_val,
            "b_y": by_val,
            "b_z": bz_val,
            "lat": lat_val,
            "lon": lon_val,
            "alt": alt_val,
            "thetax": thetax_val,
            "thetay": thetay_val,
            "thetaz": thetaz_val,
            "sensor_id": sensor_id
        }
        # --- END API Payload ---

        # --- DB Payload ---
        db_payload = {
            "timestamp": datetime.now(timezone.utc), # For MySQL DATETIME/TIMESTAMP
            "b_x": bx_val,
            "b_y": by_val,
            "b_z": bz_val,
            "lat": lat_val,
            "lon": lon_val,
            "alt": alt_val,
            "thetax": thetax_val,
            "thetay": thetay_val,
            "thetaz": thetaz_val,
            "sensor_id": sensor_id
        }
        if self.db_writer_thread and self.db_writer_thread.running:
            self.send_data_to_db_writer.emit(db_payload)
        # --- END DB Payload ---

        if self.log_file:
            # current_timestamp, bx, by, bz, lat, lon, alt, thetax, thetay, thetaz, sensor_id
            self.log_file.write(f"{current_time_pc},{bx_val},{by_val},{bz_val},{lat_val},{lon_val},{alt_val},{thetax_val},{thetay_val},{thetaz_val},{sensor_id}\n")
            self.log_file.flush()
        
        self.data_buffer['current_time_pc'].append(time.time() - self.counter)
        self.data_buffer['b_x'].append(bx_val)
        self.data_buffer['b_y'].append(by_val)
        self.data_buffer['b_z'].append(bz_val)
        self.data_buffer['lat'].append(lat_val)
        self.data_buffer['lon'].append(lon_val)
        self.data_buffer['alt'].append(alt_val)
        self.data_buffer['thetax'].append(thetax_val)
        self.data_buffer['thetay'].append(thetay_val)
        self.data_buffer['thetaz'].append(thetaz_val)
        self.data_buffer['sensor_id'].append(sensor_id)

        if len(self.data_buffer['current_time_pc']) > self.max_points:
            for key in self.data_buffer:
                self.data_buffer[key] = self.data_buffer[key][-self.max_points:]

        tbuff = self.data_buffer['current_time_pc']
        self.curve_x.setData(tbuff, self.data_buffer['b_x'])
        self.curve_y.setData(tbuff, self.data_buffer['b_y'])
        self.curve_z.setData(tbuff, self.data_buffer['b_z'])

        self.plot_widget.autoRange()

    def send_data_to_api(self):
        if self.latest_api_data is None:
            return

        payload = self.latest_api_data
        print(f"Attempting to send single data point to API: timestamp={payload.get('timestamp', 'N/A')}, b_x={payload.get('b_x', 'N/A')}...")
        threading.Thread(target=self._perform_api_post, args=(payload,)).start()

    def _perform_api_post(self, payload_dict):
        try:
            response = requests.post(WEB_API_URL, json=payload_dict, timeout=5)
            response.raise_for_status()
            message = f"API Post Successful: Sent timestamp={payload_dict.get('timestamp', 'N/A')}. Status {response.status_code}"
            print(message)
            self.api_post_result.emit(True, message)
        except requests.exceptions.Timeout:
            message = f"API Post Failed: Timeout for point timestamp={payload_dict.get('timestamp', 'N/A')}."
            print(message)
            self.api_post_result.emit(False, message)
        except requests.exceptions.ConnectionError as e:
            message = f"API Post Failed: Connection error: {e}"
            print(message)
            self.api_post_result.emit(False, message)
        except requests.exceptions.HTTPError as e:
            message = f"API Post Failed: HTTP Error {e.response.status_code} - {e.response.text}"
            print(message)
            self.api_post_result.emit(False, message)
        except Exception as e:
            message = f"API Post Failed: An unexpected error occurred: {e}"
            print(message)
            self.api_post_result.emit(False, message)

    def handle_api_post_result(self, success, message):
        self.status_label.setText(f"Status: {message}")

    def reset_plot_view(self):
        if self.data_buffer['t']:
            self.plot_widget.autoRange()
        else:
            self.plot_widget.setXRange(0, 100)
            self.plot_widget.setYRange(-10, 10)
            self.plot_widget.setTitle("Real-time Sensor Data (No Data)")
            self.status_label.setText("Plot view reset. No data available.")

    def handle_error(self, message):
        self.status_label.setText(f"Error - {message}")
        # The stop_serial() call here will stop both serial and db threads
        self.stop_serial()

    def closeEvent(self, event):
        self.stop_serial()
        event.accept()

if __name__ == "__main__":
    # Usage
    subprocess.Popen(['D:\Joy\Projects\Python311\python', 'real_time_file_watcher_db_updater.py'])
    print('opened process', ['D:\Joy\Projects\Python311\python', 'real_time_file_watcher_db_updater.py'])
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())