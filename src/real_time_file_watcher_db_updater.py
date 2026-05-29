'''
Created on Jun 17, 2025

@author: Admin
'''

print('started ok')
import pandas as pd
import mysql.connector
from mysql.connector import Error
import time
import os, sys, glob
import re
from datetime import datetime
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csv_db_updater.log'),
        logging.StreamHandler()
    ]
)

class CSVDatabaseUpdater:
    def __init__(self, csv_file_path, db_config, table_name):
        self.csv_file_path = csv_file_path
        self.db_config = db_config
        self.table_name = table_name
        self.last_file_size = 0
        self.connection = None
        self.cursor = None
        
        # Initialize database connection
        self.connect_to_database()
    
    def connect_to_database(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            logging.info("Database connection established successfully")
        except Error as e:
            logging.error(f"Error connecting to database: {e}")
            raise
    
    def reconnect_database(self):
        """Reconnect to database if connection is lost"""
        try:
            if self.connection:
                self.connection.close()
            self.connect_to_database()
            logging.info("Database reconnected successfully")
        except Error as e:
            logging.error(f"Error reconnecting to database: {e}")
            raise
    
    def read_new_data(self):
        """Read only new data from CSV file"""
        try:
            # Check if file exists
            if not os.path.exists(self.csv_file_path):
                logging.warning(f"CSV file {self.csv_file_path} does not exist")
                return []
            
            # Check file size
            current_file_size = os.path.getsize(self.csv_file_path)
            
            # If file hasn't grown, return empty list
            if current_file_size <= self.last_file_size:
                return []
            
            # Read the entire file and get only new rows
            df = pd.read_csv(self.csv_file_path)
            
            # Calculate number of rows to skip based on file size difference
            if self.last_file_size == 0:
                # First time reading, get all data
                new_rows = df
            else:
                # Calculate approximate number of new rows
                # This is a simple approach - you might need to adjust based on your specific needs
                with open(self.csv_file_path, 'r') as f:
                    f.seek(self.last_file_size)
                    new_content = f.read()
                    new_lines = new_content.strip().split('\n')
                    new_lines = [line for line in new_lines if line.strip()]  # Remove empty lines
                    
                    if len(new_lines) > 0:
                        # Parse new lines manually
                        new_data = []
                        headers = df.columns.tolist()
                        
                        for line in new_lines:
                            values = line.split(',')
                            if len(values) == len(headers):
                                row_dict = dict(zip(headers, values))
                                new_data.append(row_dict)
                        
                        new_rows = pd.DataFrame(new_data) if new_data else pd.DataFrame()
                    else:
                        new_rows = pd.DataFrame()
            
            # Update last file size
            self.last_file_size = current_file_size
            
            return new_rows
            
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            return []
    
    def process_and_insert_data(self, df):
        """Process dataframe and insert into database"""
        if df.empty:
            return
        
        try:
            # Prepare SQL query
            sql = f"""
                INSERT INTO {self.table_name}
                (b_x, b_y, b_z, timestamp, lat, lon, alt, theta_x, theta_y, theta_z, sensor_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Prepare data for insertion
            values_to_insert = []
            for _, row in df.iterrows():
                try:
                    # Parse timestamp
                    timestamp = pd.to_datetime(row['timestamp_pc']).strftime('%Y-%m-%d %H:%M:%S')
                    
                    values_to_insert.append((
                        float(row['b_x']),
                        float(row['b_y']),
                        float(row['b_z']),
                        timestamp,
                        float(row['lat']),
                        float(row['lon']),
                        float(row['alt']),
                        float(row['thetax']),
                        float(row['thetay']),
                        float(row['thetaz']),
                        str(row['sensor_id'])
                    ))
                except Exception as e:
                    logging.warning(f"Error processing row: {e}, Row data: {row.to_dict()}")
                    continue
            
            if values_to_insert:
                # Execute batch insert
                self.cursor.executemany(sql, values_to_insert)
                self.connection.commit()
                logging.info(f"Inserted {len(values_to_insert)} new records into database")
            
        except Error as e:
            logging.error(f"Database error: {e}")
            # Try to reconnect
            try:
                self.reconnect_database()
            except:
                logging.error("Failed to reconnect to database")
                raise
        except Exception as e:
            logging.error(f"Error processing data: {e}")
    
    def run_continuous_monitoring(self, poll_interval=5):
        """Main loop for continuous monitoring"""
        logging.info(f"Starting continuous monitoring of {self.csv_file_path}")
        logging.info(f"Poll interval: {poll_interval} seconds")
        
        try:
            while True:
                try:
                    # Read new data
                    new_data = self.read_new_data()
                    
                    # Process and insert if there's new data
                    if type(new_data) == list:
                        pass
                    elif not new_data.empty:
                        self.process_and_insert_data(new_data)
                    
                    # Wait before next poll
                    time.sleep(poll_interval)
                    
                except KeyboardInterrupt:
                    logging.info("Monitoring stopped by user")
                    break
                except Exception as e:
                    logging.error(f"Error in monitoring loop: {e}")
                    time.sleep(poll_interval)  # Wait before retrying
                    
        finally:
            # Clean up connections
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logging.info("Database connections closed")

# Configuration
DB_CONFIG = {
    'host': '208.109.66.54',
    'user': 'magnavis',
    'password': 'magnavis@iitk123',
    'database': 'kts-webdb-dev',
    'port': 3306
}

MAGNETIC_DATA_TABLE = "KTSWebsite_magneticdatamodel"
#
# CSV_FILE_PATH = "sensor_data_20250619_122124.csv"  # Update this path to your actual CSV file
#
# if len(sys.argv)>1:
#     CSV_FILE_PATH = sys.argv[1]

folder_dir = os.path.dirname(__file__)
max_retries = 100

CSV_FILE_PATH = ''

while (max_retries>0):
    print('folder_dir', folder_dir)
    paths = glob.glob(f'{folder_dir}/*.csv')
    latest_file_name = max(paths, key=os.path.getmtime)
    # Extract datetime string from filename using regex
    match = re.search(r'sensor_data_(\d{8}_\d{6})\.csv', latest_file_name)
    
    if match:
        file_datetime_str = match.group(1)
        # Convert to datetime object
        file_datetime = datetime.strptime(file_datetime_str, "%Y%m%d_%H%M%S")
    else:
        file_datetime = None
    
    # Get current datetime
    current_datetime = datetime.now()
    
    # Check if file was generated in the last 10 seconds
    if file_datetime:
        time_diff = (current_datetime - file_datetime).total_seconds()
        print('time diff', time_diff)
        is_recent = time_diff <= 10 and time_diff >= 0
    else:
        is_recent = False
    
    if is_recent:
        CSV_FILE_PATH = latest_file_name
        print('recent csv file found', latest_file_name)
        break
    time.sleep(1)
    
if CSV_FILE_PATH:
    print('CSV_FILE_PATH', CSV_FILE_PATH)
else:
    print('Could not find a recent file to log... exiting')
    sys.exit(0)

# Main execution
if __name__ == "__main__":
    try:
        # Create updater instance
        updater = CSVDatabaseUpdater(CSV_FILE_PATH, DB_CONFIG, MAGNETIC_DATA_TABLE)
        
        # Start continuous monitoring (polls every 5 seconds)
        updater.run_continuous_monitoring(poll_interval=5)
        
    except Exception as e:
        logging.error(f"Application error: {e}")