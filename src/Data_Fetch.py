#!/usr/bin/env python3
"""
Streamlit App for Fetching Magnetic Field Data from Database
------------------------------------------------------------
Clean, simplified version for fetching magnetic field data for specific time periods.
"""

import streamlit as st
import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
import time
import io
import logging
import sys
import traceback
from mysql.connector import pooling
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Configure logging to both file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('get_data.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Helper function to log user actions
def log_user_action(action, details=None):
    """Log user actions with timestamp"""
    msg = f"👤 USER ACTION: {action}"
    if details:
        msg += f" | Details: {details}"
    logger.info(msg)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# Helper function to log database operations
def log_db_operation(operation, query=None, params=None, result=None, error=None):
    """Log database operations with full details"""
    msg = f"🔌 DB OPERATION: {operation}"
    if query:
        msg += f" | Query: {query[:200]}"  # Limit query length
    if params:
        msg += f" | Params: {params}"
    if result:
        msg += f" | Result: {result}"
    if error:
        msg += f" | ❌ ERROR: {error}"
        logger.error(msg)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    else:
        logger.info(msg)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# Try to import SQLAlchemy for potentially faster queries
try:
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    st.warning("SQLAlchemy not available. Install with: pip install sqlalchemy pymysql")

# Page configuration
st.set_page_config(
    page_title="Magnetic Data Fetcher",
    page_icon="🧲",
    layout="wide"
)

# Database configuration
DB_CONFIG = {
    'host': '50.63.129.30',
    'user': 'devuser',
    'password': 'devuser@221133',
    'database': 'dbqnaviitk',
    'port': 3306,
    'connection_timeout': 60,
    'read_timeout': 7200,  # Extended to 2 hours for very large queries
    'write_timeout': 7200,  # Extended to 2 hours for very large queries
    'use_pure': True,
    'buffered': False,
    'autocommit': False,
    'pool_reset_session': True
}

# Configuration constants
TIME_WINDOW_MINUTES = 1  # Reduced to 1 minute to avoid query timeouts
MAX_RETRIES = 3
BASE_BACKOFF = 2
MAX_BACKOFF = 30
# Smaller fetch batches to reduce server-side timeout risk on huge pulls
CHUNK_SIZE = 20000
# Per-query row cap; keep reasonable to avoid excessive single-window payloads
MAX_ROWS_PER_WINDOW = 50000
# ID chunk sizing tuned for long-running pulls; we will auto-split if needed
ID_CHUNK_SIZE = 500000
MIN_ID_SUBCHUNK = 20000  # Minimum ID span before we stop splitting
MAX_CHUNK_SPLIT_DEPTH = 5
MAX_PARALLEL_WORKERS = 4  # Number of parallel connections (reduced to avoid pool exhaustion)
USE_PARALLEL_FETCHING = True  # Enabled - parallel chunk fetching for speed
USE_SQLALCHEMY = False  # Disabled by default - can cause issues with parallel execution
USE_PANDAS_READ_SQL = False  # Disabled by default - use proven mysql.connector method
LONG_NET_TIMEOUT = 7200  # 2 hours for net_read/net_write timeouts

# Connection pool configuration
POOL_CONFIG = {
    'pool_name': 'magnetic_data_pool',
    'pool_size': 10,  # Increased to support parallel fetching
    'pool_reset_session': True,
    **DB_CONFIG
}

TABLE_NAME = "qnav_magneticdatamodel"

# Initialize session state
if 'fetched_data' not in st.session_state:
    st.session_state.fetched_data = None
if 'fetch_status' not in st.session_state:
    st.session_state.fetch_status = None


@st.cache_resource
def get_connection_pool():
    """Create and cache a connection pool"""
    try:
        return pooling.MySQLConnectionPool(**POOL_CONFIG)
    except Exception as e:
        # Don't show error here, just return None - will fallback to direct connection
        return None


@st.cache_resource
def get_sqlalchemy_engine():
    """Create and cache a SQLAlchemy engine for faster queries (if available)"""
    if not SQLALCHEMY_AVAILABLE:
        return None
    try:
        # Use pymysql as the driver for MySQL
        connection_string = (
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            f"?charset=utf8mb4&connect_timeout={DB_CONFIG['connection_timeout']}"
        )
        engine = create_engine(
            connection_string,
            pool_size=3,
            max_overflow=5,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        # If SQLAlchemy fails, return None and fallback to mysql.connector
        return None


def ping_connection(conn):
    """Ping connection to check if it's alive"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        return True
    except:
        return False


def get_connection():
    """Get a connection from the pool or create a new one"""
    log_db_operation("get_connection", "Attempting to get connection")
    # Try pool first
    pool = get_connection_pool()
    if pool:
        try:
            log_db_operation("get_connection", "Getting connection from pool")
            conn = pool.get_connection()
            # Validate connection is alive
            if not ping_connection(conn):
                conn.close()
                log_db_operation("get_connection", error="Connection from pool is not alive")
                raise Exception("Connection from pool is not alive")
            
            cursor = conn.cursor()
            # Set longer timeouts to prevent connection loss
            cursor.execute("SET SESSION wait_timeout=28800")  # 8 hours
            cursor.execute("SET SESSION interactive_timeout=28800")  # 8 hours
            cursor.execute(f"SET SESSION net_read_timeout={LONG_NET_TIMEOUT}")  # Extended to allow very long fetches
            cursor.execute(f"SET SESSION net_write_timeout={LONG_NET_TIMEOUT}")  # Extended to allow very long fetches
            # Try to set max_execution_time (only available in MySQL 5.7.8+)
            try:
                cursor.execute("SET SESSION max_execution_time=0")  # No query timeout
            except mysql.connector.Error:
                pass  # Ignore if not supported
            cursor.close()
            log_db_operation("get_connection", result="SUCCESS - Connection from pool")
            return conn
        except Exception as pool_err:
            # Pool failed, try direct connection
            log_db_operation("get_connection", error=f"Pool connection failed: {pool_err}, trying direct connection")
            pass
    
    # Fallback to direct connection
    try:
        log_db_operation("get_connection", params=DB_CONFIG.get('host'), result="Creating direct connection")
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        # Set longer timeouts to prevent connection loss
        cursor.execute("SET SESSION wait_timeout=28800")  # 8 hours
        cursor.execute("SET SESSION interactive_timeout=28800")  # 8 hours
        cursor.execute(f"SET SESSION net_read_timeout={LONG_NET_TIMEOUT}")  # Extended to allow very long fetches
        cursor.execute(f"SET SESSION net_write_timeout={LONG_NET_TIMEOUT}")  # Extended to allow very long fetches
        # Try to set max_execution_time (only available in MySQL 5.7.8+)
        try:
            cursor.execute("SET SESSION max_execution_time=0")  # No query timeout
        except mysql.connector.Error:
            pass  # Ignore if not supported
        cursor.close()
        log_db_operation("get_connection", result="SUCCESS - Direct connection")
        return conn
    except mysql.connector.Error as db_err:
        # Re-raise database errors with full details
        error_msg = f"Database connection failed: {db_err}"
        log_db_operation("get_connection", error=error_msg)
        raise mysql.connector.Error(error_msg)
    except Exception as e:
        # Re-raise other errors
        error_msg = f"Connection error: {e}"
        log_db_operation("get_connection", error=error_msg)
        raise Exception(error_msg)


def test_connection():
    """Test database connection"""
    log_db_operation("test_connection", "Starting connection test")
    conn = None
    cursor = None
    try:
        log_db_operation("test_connection", "Getting connection from pool")
        conn = get_connection()
        if conn is None:
            error_msg = "Failed to establish connection - conn is None"
            log_db_operation("test_connection", error=error_msg)
            return False, error_msg
        
        log_db_operation("test_connection", result="Connection obtained, executing test query")
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()  # Must fetch result before closing
        cursor.close()
        cursor = None
        
        conn.close()
        conn = None
        
        if result:
            log_db_operation("test_connection", result="SUCCESS - Connection test passed")
            return True, "Connection successful"
        else:
            error_msg = "Connection test query returned no result"
            log_db_operation("test_connection", error=error_msg)
            return False, error_msg
    except mysql.connector.Error as db_err:
        error_msg = str(db_err)
        error_code = db_err.errno if hasattr(db_err, 'errno') else None
        log_db_operation("test_connection", error=f"MySQL Error {error_code}: {error_msg}")
        logger.error(traceback.format_exc())
        # Clean up
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if conn:
            try:
                conn.close()
            except:
                pass
        
        # Provide more helpful error messages
        if "Access denied" in error_msg or "1045" in error_msg:
            return False, f"Authentication failed: {error_msg}"
        elif "Can't connect" in error_msg or "2003" in error_msg:
            return False, f"Network error - cannot reach database server: {error_msg}"
        elif "Unknown database" in error_msg or "1049" in error_msg:
            return False, f"Database not found: {error_msg}"
        else:
            return False, f"Database error: {error_msg}"
    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        log_db_operation("test_connection", error=error_msg)
        logger.error(traceback.format_exc())
        # Clean up
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if conn:
            try:
                conn.close()
            except:
                pass
        return False, f"Connection failed: {error_msg}"


@st.cache_data(ttl=600)
def get_available_sensors():
    """Get list of available sensors from database"""
    log_db_operation("get_available_sensors", "Starting")
    conn = None
    cursor = None
    try:
        # Use direct connection with extended timeouts
        log_db_operation("get_available_sensors", params=DB_CONFIG.get('host'), result="Attempting connection")
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn is None:
            error_msg = "Failed to get database connection - conn is None"
            log_db_operation("get_available_sensors", error=error_msg)
            return []
        
        log_db_operation("get_available_sensors", "Connection established successfully")
        
        # Set timeouts on this connection
        setup_cursor = conn.cursor()
        try:
            setup_cursor.execute("SET SESSION wait_timeout=28800")
            setup_cursor.execute("SET SESSION interactive_timeout=28800")
            setup_cursor.execute(f"SET SESSION net_read_timeout={LONG_NET_TIMEOUT}")
            setup_cursor.execute(f"SET SESSION net_write_timeout={LONG_NET_TIMEOUT}")
            log_db_operation("get_available_sensors", "Session timeouts set")
        except Exception as timeout_err:
            log_db_operation("get_available_sensors", error=f"Failed to set timeouts: {timeout_err}")
        finally:
            setup_cursor.close()
        
        query = f"SELECT DISTINCT sensor_id FROM {TABLE_NAME} ORDER BY sensor_id"
        log_db_operation("get_available_sensors", query=query)
        
        cursor = conn.cursor()
        cursor.execute(query)
        all_rows = cursor.fetchall()
        sensors = [row[0] for row in all_rows if row[0] is not None]
        
        log_db_operation("get_available_sensors", query=query, result=f"Found {len(sensors)} sensors: {sensors}")
        
        cursor.close()
        cursor = None
        conn.close()
        conn = None
        return sensors if sensors else []
    except mysql.connector.Error as db_err:
        error_msg = f"MySQL Error {db_err.errno}: {str(db_err)}"
        log_db_operation("get_available_sensors", error=error_msg)
        logger.error(traceback.format_exc())
        # Clean up on error
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if conn:
            try:
                conn.close()
            except:
                pass
        return []
    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        log_db_operation("get_available_sensors", error=error_msg)
        logger.error(traceback.format_exc())
        # Clean up on error
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if conn:
            try:
                conn.close()
            except:
                pass
        return []


@st.cache_data(ttl=600)
def get_data_range():
    """Get the date range of available data"""
    log_db_operation("get_data_range", "Starting")
    conn = None
    cursor = None
    try:
        # Use direct connection with extended timeouts
        log_db_operation("get_data_range", params=DB_CONFIG.get('host'), result="Attempting connection")
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn is None:
            error_msg = "Failed to get database connection - conn is None"
            log_db_operation("get_data_range", error=error_msg)
            raise Exception(error_msg)
        
        log_db_operation("get_data_range", "Connection established successfully")
        
        # Set timeouts on this connection
        setup_cursor = conn.cursor()
        try:
            setup_cursor.execute("SET SESSION wait_timeout=28800")
            setup_cursor.execute("SET SESSION interactive_timeout=28800")
            setup_cursor.execute(f"SET SESSION net_read_timeout={LONG_NET_TIMEOUT}")
            setup_cursor.execute(f"SET SESSION net_write_timeout={LONG_NET_TIMEOUT}")
            log_db_operation("get_data_range", "Session timeouts set")
        except Exception as timeout_err:
            log_db_operation("get_data_range", error=f"Failed to set timeouts: {timeout_err}")
        finally:
            setup_cursor.close()
        
        query = f"SELECT MIN(timestamp), MAX(timestamp) FROM {TABLE_NAME}"
        log_db_operation("get_data_range", query=query)
        
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        
        log_db_operation("get_data_range", query=query, result=f"MIN: {result[0] if result and result[0] else 'None'}, MAX: {result[1] if result and result[1] else 'None'}")
        
        cursor.close()
        cursor = None
        conn.close()
        conn = None
        
        if result and result[0] and result[1]:
            log_db_operation("get_data_range", result="SUCCESS", details=f"Range: {result[0]} to {result[1]}")
            return result[0], result[1]
        
        log_db_operation("get_data_range", result="No data found (result is None or empty)")
        return None, None
    except mysql.connector.Error as db_err:
        error_msg = f"MySQL Error {db_err.errno}: {str(db_err)}"
        log_db_operation("get_data_range", error=error_msg)
        logger.error(traceback.format_exc())
        # Clean up on error
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if conn:
            try:
                conn.close()
            except:
                pass
        st.error(f"Error fetching data range: {error_msg}")
        return None, None
    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        log_db_operation("get_data_range", error=error_msg)
        logger.error(traceback.format_exc())
        # Clean up on error
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if conn:
            try:
                conn.close()
            except:
                pass
        st.error(f"Error fetching data range: {error_msg}")
        return None, None


def get_table_id_bounds():
    """Get the min and max ID in the table - very fast query"""
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        # This is instant - uses primary key index
        cursor.execute(f"SELECT MIN(id), MAX(id) FROM {TABLE_NAME}")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result and result[0] is not None and result[1] is not None:
            return result[0], result[1]
        return None, None
    except Exception as e:
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if conn:
            try:
                conn.close()
            except:
                pass
        return None, None


def find_id_for_timestamp(target_time, min_id, max_id, sensor_ids=None, is_start=True):
    """Find approximate ID for a timestamp using linear interpolation on sampled points"""
    log_db_operation("find_id_for_timestamp", result=f"Finding ID for {target_time} (is_start={is_start})")
    conn = None
    cursor = None
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Sample 10 evenly-spaced IDs across the range to build time-to-ID mapping
        num_samples = 10
        sample_ids = []
        id_range = max_id - min_id
        step = max(1, id_range // num_samples)
        
        for i in range(num_samples + 1):
            sample_id = min_id + (i * step)
            sample_id = min(max_id, max(min_id, sample_id))
            sample_ids.append(sample_id)
        
        # Remove duplicates and sort
        sample_ids = sorted(list(set(sample_ids)))
        
        # Query timestamps for these sample IDs (very fast - uses primary key index)
        placeholders = ','.join(['%s'] * len(sample_ids))
        query = f"SELECT id, timestamp FROM {TABLE_NAME} WHERE id IN ({placeholders})"
        params = sample_ids
        
        if sensor_ids and len(sensor_ids) > 0:
            sensor_placeholders = ','.join(['%s'] * len(sensor_ids))
            query += f" AND sensor_id IN ({sensor_placeholders})"
            params.extend(sensor_ids)
        
        query += " ORDER BY id"
        
        cursor.execute(query, params)
        samples = cursor.fetchall()
        
        if not samples or len(samples) < 2:
            # Not enough samples, use simple midpoint
            fallback_id = (min_id + max_id) // 2
            log_db_operation("find_id_for_timestamp", result=f"Not enough samples, using fallback ID: {fallback_id}")
            cursor.close()
            conn.close()
            return fallback_id
        
        # Find the two samples that bracket target_time
        best_id = None
        for i in range(len(samples) - 1):
            id1, time1 = samples[i]
            id2, time2 = samples[i + 1]
            
            if time1 is None or time2 is None:
                continue
            
            # Check if target_time is between these two samples
            if time1 <= target_time <= time2:
                # Linear interpolation
                if time2 == time1:
                    best_id = id1
                else:
                    # Interpolate ID based on time ratio
                    time_ratio = (target_time - time1).total_seconds() / (time2 - time1).total_seconds()
                    best_id = int(id1 + (id2 - id1) * time_ratio)
                break
            elif is_start and time1 > target_time:
                # target_time is before all samples, use first sample
                best_id = id1
                break
            elif not is_start and time2 < target_time:
                # target_time is after all samples, use last sample
                best_id = id2
        
        cursor.close()
        conn.close()
        
        if best_id:
            # Ensure ID is within bounds
            best_id = max(min_id, min(max_id, best_id))
            log_db_operation("find_id_for_timestamp", result=f"Found approximate ID: {best_id}")
            return best_id
        
        # Fallback: use midpoint
        fallback_id = (min_id + max_id) // 2
        log_db_operation("find_id_for_timestamp", result=f"Using fallback ID: {fallback_id}")
        return fallback_id
        
    except Exception as e:
        log_db_operation("find_id_for_timestamp", error=f"Exception: {str(e)}")
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if conn:
            try:
                conn.close()
            except:
                pass
        # Return safe fallback
        return (min_id + max_id) // 2


def find_id_range_for_time_period(start_time, end_time, sensor_ids=None):
    """Find the ID range for a time period using fast binary search"""
    log_db_operation("find_id_range", result=f"Starting - Start: {start_time}, End: {end_time}")
    
    try:
        # Step 1: Get table bounds (instant query)
        min_id, max_id = get_table_id_bounds()
        if min_id is None or max_id is None:
            log_db_operation("find_id_range", error="Could not get table ID bounds")
            return None, None
        
        log_db_operation("find_id_range", result=f"Table ID range: {min_id:,} to {max_id:,}")
        
        # Step 2: Find approximate IDs for start and end times using interpolation
        start_id_approx = find_id_for_timestamp(start_time, min_id, max_id, sensor_ids, is_start=True)
        end_id_approx = find_id_for_timestamp(end_time, min_id, max_id, sensor_ids, is_start=False)
        
        # Ensure start_id <= end_id
        if start_id_approx > end_id_approx:
            start_id_approx, end_id_approx = end_id_approx, start_id_approx
        
        # Step 3: Refine using actual MIN/MAX queries on a narrowed range (faster than full table)
        # Use a search window around the estimated IDs
        search_window = max(100000, (end_id_approx - start_id_approx) * 2)  # At least 100k IDs on each side
        refined_min_id = max(min_id, start_id_approx - search_window)
        refined_max_id = min(max_id, end_id_approx + search_window)
        
        # Try refinement with timeout to avoid hanging
        start_id = None
        end_id = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            # Set a query timeout (5 seconds for refinement)
            cursor = conn.cursor()
            try:
                cursor.execute("SET SESSION max_execution_time=5000")  # 5 seconds
            except:
                pass  # Ignore if not supported
            
            # Get actual MIN/MAX IDs in the refined range (much faster than full table)
            query = f"SELECT MIN(id), MAX(id) FROM {TABLE_NAME} WHERE id >= %s AND id <= %s AND timestamp >= %s AND timestamp <= %s"
            params = [refined_min_id, refined_max_id, start_time, end_time]
            
            if sensor_ids and len(sensor_ids) > 0:
                placeholders = ','.join(['%s'] * len(sensor_ids))
                query += f" AND sensor_id IN ({placeholders})"
                params.extend(sensor_ids)
            
            log_db_operation("find_id_range", query=query[:200], result=f"Refining ID range with actual MIN/MAX query (window: {refined_min_id:,} to {refined_max_id:,})")
            
            # Execute refinement query (with timeout set above)
            try:
                cursor.execute(query, params)
                result = cursor.fetchone()
                
                if result and result[0] is not None and result[1] is not None:
                    # Use the refined IDs
                    start_id = result[0]
                    end_id = result[1]
                    log_db_operation("find_id_range", result=f"Refined ID range: {start_id:,} to {end_id:,}")
                else:
                    log_db_operation("find_id_range", result="Refinement returned None - no data in refined range")
            except Exception as query_err:
                log_db_operation("find_id_range", error=f"Refinement query failed: {query_err}")
            
            cursor.close()
            conn.close()
        except Exception as refine_err:
            # If refinement fails, log and continue to fallback
            log_db_operation("find_id_range", error=f"Refinement failed: {refine_err}, using approximate IDs")
        
        # Fallback to approximate IDs if refinement didn't work
        if start_id is None or end_id is None:
            id_range = end_id_approx - start_id_approx
            # Use a larger margin (5%) if refinement failed, to ensure we don't miss data
            start_id = max(min_id, int(start_id_approx - id_range * 0.05))
            end_id = min(max_id, int(end_id_approx + id_range * 0.05))
            log_db_operation("find_id_range", result=f"Using approximate IDs (refinement unavailable): {start_id:,} to {end_id:,}")
        
        log_db_operation("find_id_range", result=f"SUCCESS: Final ID range {start_id:,} to {end_id:,}")
        return start_id, end_id
        
    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        log_db_operation("find_id_range", error=error_msg)
        logger.error(traceback.format_exc())
        return None, None


def fetch_single_window(window_start, window_end, columns_str, sensor_ids, downsample, window_num, total_windows):
    """
    Fetch data for a single time window - optimized for parallel execution
    Returns (window_num, rows_list) or (window_num, None) on failure
    """
    log_db_operation(f"fetch_single_window_{window_num}", 
                     params=f"Start: {window_start}, End: {window_end}",
                     result=f"Starting window {window_num}/{total_windows}")
    retries = 0
    conn = None
    cursor = None
    
    while retries < MAX_RETRIES:
        try:
            # Get fresh connection for each window (important for parallel execution)
            # Don't use pool for parallel execution - create direct connections
            log_db_operation(f"fetch_single_window_{window_num}", result=f"Creating connection (attempt {retries + 1})")
            conn = mysql.connector.connect(**DB_CONFIG)
            if conn is None:
                raise Exception("Failed to get database connection")
            
            log_db_operation(f"fetch_single_window_{window_num}", result="Connection created, setting timeouts")
            # Set timeouts on this connection
            cursor = conn.cursor()
            cursor.execute("SET SESSION wait_timeout=28800")
            cursor.execute("SET SESSION interactive_timeout=28800")
            cursor.execute(f"SET SESSION net_read_timeout={LONG_NET_TIMEOUT}")
            cursor.execute(f"SET SESSION net_write_timeout={LONG_NET_TIMEOUT}")
            cursor.close()
            
            # Validate connection is alive
            if not ping_connection(conn):
                conn.close()
                raise Exception("Connection is not alive")
            
            log_db_operation(f"fetch_single_window_{window_num}", result="Connection validated, building query")
            cursor = conn.cursor(dictionary=True, buffered=False)
            
            # Build query
            query = f"SELECT {columns_str} FROM {TABLE_NAME} WHERE timestamp >= %s AND timestamp <= %s"
            params = [window_start, window_end]
            
            if sensor_ids and len(sensor_ids) > 0:
                placeholders = ','.join(['%s'] * len(sensor_ids))
                query += f" AND sensor_id IN ({placeholders})"
                params.extend(sensor_ids)
            
            if downsample and downsample > 1:
                query += f" AND MOD(id, {downsample}) = 0"
            
            # No ORDER BY for speed
            query += f" LIMIT {MAX_ROWS_PER_WINDOW}"
            
            log_db_operation(f"fetch_single_window_{window_num}", query=query[:150], result="Executing query")
            start_exec_time = time.time()
            cursor.execute(query, params)
            exec_time = time.time() - start_exec_time
            log_db_operation(f"fetch_single_window_{window_num}", result=f"Query executed in {exec_time:.2f}s, starting fetch")
            
            window_rows = []
            rows_fetched = 0
            fetch_start_time = time.time()
            while True:
                fetch_chunk_start = time.time()
                rows = cursor.fetchmany(CHUNK_SIZE)
                fetch_chunk_time = time.time() - fetch_chunk_start
                
                if not rows:
                    break
                window_rows.extend(rows)
                rows_fetched += len(rows)
                
                # Log progress every 10k rows
                if rows_fetched % 10000 == 0:
                    elapsed = time.time() - fetch_start_time
                    log_db_operation(f"fetch_single_window_{window_num}", 
                                    f"Progress: {rows_fetched:,} rows fetched in {elapsed:.2f}s")
                
                # Keep connection alive during long fetches - more frequent keepalive
                if rows_fetched % 2000 == 0:  # More frequent keepalive (every 2k rows instead of 5k)
                    try:
                        # Use a lightweight keepalive - just execute a simple query
                        keepalive_cursor = conn.cursor()
                        keepalive_cursor.execute("SELECT 1")
                        keepalive_cursor.fetchone()
                        keepalive_cursor.close()
                        log_db_operation(f"fetch_single_window_{window_num}", 
                                        f"Keepalive ping successful at {rows_fetched:,} rows")
                    except Exception as keepalive_err:
                        # If keepalive fails, connection is lost
                        error_msg = f"Connection lost during fetch at {rows_fetched:,} rows: {keepalive_err}"
                        log_db_operation(f"fetch_single_window_{window_num}", error=error_msg)
                        raise mysql.connector.Error(error_msg)
            
            total_time = time.time() - fetch_start_time
            log_db_operation(f"fetch_single_window_{window_num}", 
                           f"SUCCESS: Fetched {rows_fetched:,} rows in {total_time:.2f}s")
            
            cursor.close()
            cursor = None
            conn.close()
            conn = None
            
            return (window_num, window_rows)
            
        except (mysql.connector.Error, Exception) as db_err:
            retries += 1
            # Handle both mysql.connector and SQLAlchemy errors
            error_code = None
            if isinstance(db_err, mysql.connector.Error):
                error_code = db_err.errno if hasattr(db_err, 'errno') else None
            elif hasattr(db_err, 'orig') and hasattr(db_err.orig, 'errno'):
                error_code = db_err.orig.errno
            
            error_msg = str(db_err)
            log_db_operation(f"fetch_single_window_{window_num}", 
                          error=f"Attempt {retries}/{MAX_RETRIES} failed: Error {error_code}: {error_msg}")
            logger.error(traceback.format_exc())
            
            # Clean up cursor if it exists
            if 'cursor' in locals() and cursor:
                try:
                    cursor.close()
                except:
                    pass
            cursor = None
            
            # Clean up connection if it exists
            if 'conn' in locals() and conn:
                try:
                    conn.close()
                except:
                    pass
            conn = None
            
            if retries < MAX_RETRIES:
                if error_code == 2013 or "Lost connection" in error_msg or "Connection" in error_msg:
                    delay = min(BASE_BACKOFF * (2 ** retries), MAX_BACKOFF * 2)
                else:
                    delay = min(BASE_BACKOFF * (2 ** (retries - 1)), MAX_BACKOFF)
                log_db_operation(f"fetch_single_window_{window_num}", 
                               f"Retrying in {delay}s (attempt {retries}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                # Log the error for debugging
                log_db_operation(f"fetch_single_window_{window_num}", 
                               error=f"All {MAX_RETRIES} retries exhausted. Window failed.")
                return (window_num, None)
    
    # All retries exhausted
    log_db_operation(f"fetch_single_window_{window_num}", 
                   error="All retries exhausted outside retry loop")
    return (window_num, None)


def fetch_chunk(id_start, id_end, columns_str, sensor_ids, downsample, start_time, end_time, progress_container, chunk_num, total_chunks, split_depth=0):
    """Fetch data for an ID chunk using pure ID-based query (no timestamp WHERE clause)
    
    If a chunk repeatedly times out, we recursively split it into smaller sub-chunks
    (up to MAX_CHUNK_SPLIT_DEPTH) so that very large pulls can complete even if
    individual queries would otherwise exceed server timeouts.
    """
    retries = 0
    conn = None
    cursor = None
    
    while retries < MAX_RETRIES:
        try:
            # Use direct connection with extended timeouts
            conn = mysql.connector.connect(**DB_CONFIG)
            if conn is None:
                raise Exception("Failed to get database connection")
            
            # Set timeouts on this connection
            setup_cursor = conn.cursor()
            setup_cursor.execute("SET SESSION wait_timeout=28800")
            setup_cursor.execute("SET SESSION interactive_timeout=28800")
            setup_cursor.execute(f"SET SESSION net_read_timeout={LONG_NET_TIMEOUT}")
            setup_cursor.execute(f"SET SESSION net_write_timeout={LONG_NET_TIMEOUT}")
            setup_cursor.close()
            
            cursor = conn.cursor(dictionary=True, buffered=False)
            
            # HYBRID APPROACH: ID range + timestamp filter
            # ID range narrows search space (faster), timestamp filter avoids fetching irrelevant rows
            query = f"SELECT {columns_str} FROM {TABLE_NAME} WHERE id >= %s AND id <= %s AND timestamp >= %s AND timestamp <= %s"
            params = [id_start, id_end, start_time, end_time]
            
            if sensor_ids and len(sensor_ids) > 0:
                placeholders = ','.join(['%s'] * len(sensor_ids))
                query += f" AND sensor_id IN ({placeholders})"
                params.extend(sensor_ids)
            
            if downsample and downsample > 1:
                query += f" AND MOD(id, {downsample}) = 0"
            
            # No ORDER BY - use primary key index for speed
            # No LIMIT - fetch all matching rows in this ID range
            
            log_db_operation(f"fetch_chunk_{chunk_num}", query=query[:200], result=f"Executing hybrid ID+timestamp query")
            exec_start = time.time()
            cursor.execute(query, params)
            exec_time = time.time() - exec_start
            log_db_operation(f"fetch_chunk_{chunk_num}", result=f"Query executed in {exec_time:.2f}s, starting fetch")
            
            chunk_rows = []
            rows_fetched = 0
            fetch_start = time.time()
            
            while True:
                rows = cursor.fetchmany(CHUNK_SIZE)
                if not rows:
                    break
                
                # No need to filter - SQL already filtered by timestamp
                chunk_rows.extend(rows)
                rows_fetched += len(rows)
                
                # Log progress
                if rows_fetched % 10000 == 0:
                    elapsed = time.time() - fetch_start
                    log_db_operation(f"fetch_chunk_{chunk_num}", 
                                    result=f"Progress: {rows_fetched:,} rows fetched in {elapsed:.2f}s")
            
            total_time = time.time() - fetch_start
            log_db_operation(f"fetch_chunk_{chunk_num}", 
                           result=f"SUCCESS: Fetched {rows_fetched:,} rows in {total_time:.2f}s")
            
            cursor.close()
            conn.close()
            return chunk_rows
            
        except Exception as err:
            retries += 1
            error_msg = str(err)
            log_db_operation(f"fetch_chunk_{chunk_num}", error=f"Attempt {retries}/{MAX_RETRIES} failed: {error_msg}")
            
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass
            
            if retries < MAX_RETRIES:
                delay = min(BASE_BACKOFF * (2 ** (retries - 1)), MAX_BACKOFF)
                if progress_container:
                    progress_container.text(f"⚠️ Chunk {chunk_num}/{total_chunks} failed. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                log_db_operation(f"fetch_chunk_{chunk_num}", error="All retries exhausted")
                break
    
    # If we reach here, the chunk failed; attempt to split if allowed
    id_span = id_end - id_start
    if split_depth < MAX_CHUNK_SPLIT_DEPTH and id_span > MIN_ID_SUBCHUNK:
        mid_id = (id_start + id_end) // 2
        log_db_operation(
            f"fetch_chunk_{chunk_num}",
            result=(
                f"Splitting chunk due to repeated failures (depth {split_depth}/{MAX_CHUNK_SPLIT_DEPTH}). "
                f"Ranges: [{id_start}, {mid_id}] and [{mid_id + 1}, {id_end}]"
            )
        )
        left_rows = fetch_chunk(
            id_start, mid_id, columns_str, sensor_ids, downsample,
            start_time, end_time, progress_container, chunk_num,
            total_chunks, split_depth + 1
        )
        right_rows = fetch_chunk(
            mid_id + 1, id_end, columns_str, sensor_ids, downsample,
            start_time, end_time, progress_container, chunk_num,
            total_chunks, split_depth + 1
        )
        if left_rows is None and right_rows is None:
            return None
        combined = []
        if left_rows:
            combined.extend(left_rows)
        if right_rows:
            combined.extend(right_rows)
        return combined
    
    return None
    

def fetch_data(start_time, end_time, columns=None, sensor_ids=None, downsample=None, limit_rows=None):
    """Fetch data from database for specified time range"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    progress_container = st.empty()
    
    try:
        # Default columns
        if columns is None or len(columns) == 0:
            columns = ["id", "sensor_id", "timestamp", "b_x", "b_y", "b_z", "lat", "lon", "alt", "theta_x", "theta_y", "theta_z"]
        
        columns_str = ', '.join(columns)
        total_duration = (end_time - start_time).total_seconds() / 60
        
        # Always try ID-based chunking first (faster because IDs are indexed)
        use_id_chunking = False
        id_min_range = None
        id_max_range = None
        
        # Try to find ID range - this is much faster than time-based queries
        status_text.text("🔍 Finding ID range for time period (faster method)...")
        try:
            id_min_range, id_max_range = find_id_range_for_time_period(start_time, end_time, sensor_ids)
            
            if id_min_range is not None and id_max_range is not None:
                use_id_chunking = True
                status_text.text(f"✅ Found ID range: {id_min_range:,} to {id_max_range:,}")
                log_db_operation("fetch_data", result=f"ID-based chunking enabled: IDs {id_min_range:,} to {id_max_range:,}")
            else:
                log_db_operation("fetch_data", result="ID range not found, falling back to time-based windows")
        except Exception as id_range_err:
            log_db_operation("fetch_data", error=f"Failed to find ID range: {id_range_err}, falling back to time-based windows")
            status_text.text("⚠️ Could not determine ID range, using time-based windows...")
        
        # Use ID-based chunking if available (preferred method - much faster)
        if use_id_chunking and id_min_range is not None and id_max_range is not None:
            total_ids = id_max_range - id_min_range
            num_chunks = max(1, (total_ids // ID_CHUNK_SIZE) + (1 if total_ids % ID_CHUNK_SIZE > 0 else 0))
            
            # Use parallel fetching if enabled and we have multiple chunks
            if USE_PARALLEL_FETCHING and num_chunks > 1:
                status_text.text(f"🚀 Fetching {num_chunks} chunks in parallel ({MAX_PARALLEL_WORKERS} workers)...")
                progress_container.text(f"⚡ Using parallel fetching for faster performance")
                
                all_rows = []
                total_fetched = 0
                start_time_fetch = time.time()
                completed_chunks = 0
                failed_chunks = []
                results_dict = {}
                
                # Use ThreadPoolExecutor for parallel fetching
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
                    # Submit all chunk fetch tasks
                    future_to_chunk = {
                        executor.submit(
                            fetch_chunk,
                            id_min_range + (chunk_num * ID_CHUNK_SIZE),
                            min(id_min_range + ((chunk_num + 1) * ID_CHUNK_SIZE) - 1, id_max_range),
                            columns_str,
                            sensor_ids,
                            downsample,
                            start_time,
                            end_time,
                            None,  # progress_container not used in parallel mode
                            chunk_num + 1,
                            num_chunks
                        ): chunk_num
                        for chunk_num in range(num_chunks)
                    }
                    
                    # Process completed tasks as they finish
                    consecutive_empty = 0
                    max_consecutive_empty = 10  # Stop if 10 consecutive chunks are empty
                    max_empty_without_data = 20  # Stop if 20 chunks are empty and we've found no data at all
                    last_data_chunk = -1
                    total_empty = 0
                    
                    for future in as_completed(future_to_chunk):
                        chunk_num = future_to_chunk[future]
                        try:
                            chunk_rows = future.result()
                            
                            if chunk_rows is not None:
                                if len(chunk_rows) > 0:
                                    # Found data - reset empty counter
                                    consecutive_empty = 0
                                    total_empty = 0
                                    last_data_chunk = chunk_num
                                    results_dict[chunk_num] = chunk_rows
                                    completed_chunks += 1
                                    total_fetched += len(chunk_rows)
                                    
                                    progress = completed_chunks / num_chunks
                                    progress_bar.progress(progress)
                                    
                                    elapsed = time.time() - start_time_fetch
                                    if elapsed > 0:
                                        rate = total_fetched / elapsed
                                        progress_container.text(
                                            f"✅ Completed {completed_chunks}/{num_chunks} chunks | "
                                            f"Total: {total_fetched:,} rows | Rate: {rate:.0f} rows/sec"
                                        )
                                else:
                                    # Empty chunk
                                    consecutive_empty += 1
                                    total_empty += 1
                                    completed_chunks += 1
                                    log_db_operation("fetch_data", result=f"Chunk {chunk_num + 1} returned 0 rows (empty)")
                                    
                                    # Early termination scenarios:
                                    # 1. Found data before, but now getting many consecutive empty chunks
                                    if consecutive_empty >= max_consecutive_empty and last_data_chunk >= 0:
                                        log_db_operation("fetch_data", result=f"Early termination: {consecutive_empty} consecutive empty chunks after chunk {last_data_chunk + 1}")
                                        progress_container.text(
                                            f"⚠️ Early termination: {consecutive_empty} consecutive empty chunks. Data range may be complete."
                                        )
                                        # Cancel remaining futures
                                        for f in future_to_chunk:
                                            if f != future:
                                                f.cancel()
                                        break
                                    # 2. Never found any data, and processed many empty chunks
                                    elif total_empty >= max_empty_without_data and last_data_chunk < 0:
                                        log_db_operation("fetch_data", result=f"Early termination: {total_empty} empty chunks with no data found. ID range may be incorrect.")
                                        progress_container.text(
                                            f"⚠️ Early termination: {total_empty} empty chunks with no data. ID range estimation may be incorrect. Try checking data range first."
                                        )
                                        # Cancel remaining futures
                                        for f in future_to_chunk:
                                            if f != future:
                                                f.cancel()
                                        break
                            else:
                                failed_chunks.append(chunk_num + 1)
                                progress_container.text(
                                    f"❌ Chunk {chunk_num + 1}/{num_chunks} failed after retries"
                                )
                        except Exception as exc:
                            failed_chunks.append(chunk_num + 1)
                            progress_container.text(
                                f"❌ Chunk {chunk_num + 1}/{num_chunks} generated exception: {exc}"
                            )
                
                # Sort results by chunk number and combine
                for chunk_num in sorted(results_dict.keys()):
                    all_rows.extend(results_dict[chunk_num])
                
                # If too many chunks failed, return error
                if len(failed_chunks) > num_chunks * 0.5:  # More than 50% failed
                    return None, f"Failed to fetch {len(failed_chunks)} out of {num_chunks} chunks: {failed_chunks[:10]}{'...' if len(failed_chunks) > 10 else ''}"
                elif failed_chunks:
                    # Some chunks failed but not too many - warn but continue
                    log_db_operation("fetch_data", result=f"Warning: {len(failed_chunks)} chunks failed: {failed_chunks[:10]}")
            else:
                # Sequential fetching (fallback)
                status_text.text(f"🔄 Fetching data using ID-based chunks ({num_chunks} chunks)...")
                
                all_rows = []
                total_fetched = 0
                start_time_fetch = time.time()
                
                consecutive_empty = 0
                max_consecutive_empty = 10  # Stop if 10 consecutive chunks are empty
                max_empty_without_data = 20  # Stop if 20 chunks are empty and we've found no data at all
                last_data_chunk = -1
                total_empty = 0
                
                for chunk_num in range(num_chunks):
                    chunk_id_start = id_min_range + (chunk_num * ID_CHUNK_SIZE)
                    chunk_id_end = min(id_min_range + ((chunk_num + 1) * ID_CHUNK_SIZE) - 1, id_max_range)
                    
                    progress_container.text(f"🔄 Processing chunk {chunk_num + 1}/{num_chunks}: IDs {chunk_id_start:,} to {chunk_id_end:,}")
                    
                    chunk_rows = fetch_chunk(
                        chunk_id_start, chunk_id_end, columns_str, sensor_ids, downsample,
                        start_time, end_time, progress_container, chunk_num + 1, num_chunks
                    )
                    
                    if chunk_rows is not None:
                        if len(chunk_rows) > 0:
                            # Found data - reset empty counter
                            consecutive_empty = 0
                            total_empty = 0
                            last_data_chunk = chunk_num
                            all_rows.extend(chunk_rows)
                            total_fetched += len(chunk_rows)
                            
                            progress = (chunk_num + 1) / num_chunks
                            progress_bar.progress(progress)
                            
                            elapsed = time.time() - start_time_fetch
                            if elapsed > 0:
                                rate = total_fetched / elapsed
                                progress_container.text(
                                    f"✅ Chunk {chunk_num + 1}/{num_chunks} complete ({len(chunk_rows):,} rows) | "
                                    f"Total: {total_fetched:,} rows | Rate: {rate:.0f} rows/sec"
                                )
                        else:
                            # Empty chunk
                            consecutive_empty += 1
                            total_empty += 1
                            log_db_operation("fetch_data", result=f"Chunk {chunk_num + 1} returned 0 rows (empty)")
                            
                            # Early termination scenarios:
                            # 1. Found data before, but now getting many consecutive empty chunks
                            if consecutive_empty >= max_consecutive_empty and last_data_chunk >= 0:
                                log_db_operation("fetch_data", result=f"Early termination: {consecutive_empty} consecutive empty chunks after chunk {last_data_chunk + 1}")
                                progress_container.text(
                                    f"⚠️ Early termination: {consecutive_empty} consecutive empty chunks. Data range may be complete."
                                )
                                break
                            # 2. Never found any data, and processed many empty chunks
                            elif total_empty >= max_empty_without_data and last_data_chunk < 0:
                                log_db_operation("fetch_data", result=f"Early termination: {total_empty} empty chunks with no data found. ID range may be incorrect.")
                                progress_container.text(
                                    f"⚠️ Early termination: {total_empty} empty chunks with no data. ID range estimation may be incorrect. Try checking data range first."
                                )
                                break
                    else:
                        progress_container.text(f"❌ Chunk {chunk_num + 1}/{num_chunks} failed")
                        return None, f"Failed to fetch chunk {chunk_num + 1}"
        
        # Fallback to time-window approach
        elif total_duration > TIME_WINDOW_MINUTES:
            window_duration = timedelta(minutes=TIME_WINDOW_MINUTES)
            windows = []
            current_start = start_time
            
            while current_start < end_time:
                current_end = min(current_start + window_duration, end_time)
                windows.append((current_start, current_end))
                current_start = current_end
            
            total_windows = len(windows)
            
            # Use parallel fetching if enabled and we have multiple windows
            # Temporarily disabled - needs debugging
            if False and USE_PARALLEL_FETCHING and total_windows > 2:
                status_text.text(f"🚀 Fetching {total_windows} windows in parallel ({MAX_PARALLEL_WORKERS} workers)...")
                progress_container.text(f"⚡ Using parallel fetching for faster performance")
                
                all_rows = []
                total_fetched = 0
                start_time_fetch = time.time()
                completed_windows = 0
                failed_windows = []
                results_dict = {}
                
                # Use ThreadPoolExecutor for parallel fetching
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
                    # Submit all window fetch tasks
                    future_to_window = {
                        executor.submit(
                            fetch_single_window,
                            window_start,
                            window_end,
                            columns_str,
                            sensor_ids,
                            downsample,
                            window_num,
                            total_windows
                        ): (window_num, window_start, window_end)
                        for window_num, (window_start, window_end) in enumerate(windows, 1)
                    }
                    
                    # Process completed tasks as they finish
                    for future in as_completed(future_to_window):
                        window_num, window_start, window_end = future_to_window[future]
                        try:
                            result_window_num, window_rows = future.result()
                            
                            if window_rows is not None:
                                results_dict[result_window_num] = window_rows
                                completed_windows += 1
                                total_fetched += len(window_rows)
                                
                                progress = completed_windows / total_windows
                                progress_bar.progress(progress)
                                
                                elapsed = time.time() - start_time_fetch
                                if elapsed > 0:
                                    rate = total_fetched / elapsed
                                    progress_container.text(
                                        f"✅ Completed {completed_windows}/{total_windows} windows | "
                                        f"Total: {total_fetched:,} rows | Rate: {rate:.0f} rows/sec"
                                    )
                            else:
                                failed_windows.append(window_num)
                                progress_container.text(
                                    f"❌ Window {window_num}/{total_windows} failed after retries"
                                )
                        except Exception as exc:
                            failed_windows.append(window_num)
                            progress_container.text(
                                f"❌ Window {window_num}/{total_windows} generated exception: {exc}"
                            )
                
                # Sort results by window number and combine
                for window_num in sorted(results_dict.keys()):
                    all_rows.extend(results_dict[window_num])
                
                # If too many windows failed, fall back to sequential
                if len(failed_windows) > total_windows * 0.5:  # More than 50% failed
                    status_text.text(f"⚠️ Parallel fetching failed for {len(failed_windows)} windows. Falling back to sequential...")
                    progress_container.text("🔄 Retrying with sequential fetching...")
                    # Fall through to sequential method below
                elif failed_windows:
                    # Some windows failed but not too many - return error
                    return None, f"Failed to fetch {len(failed_windows)} out of {total_windows} windows: {failed_windows[:10]}{'...' if len(failed_windows) > 10 else ''}"
                else:
                    # All windows succeeded
                    pass
            
            else:
                # Sequential fetching (original method)
                status_text.text(f"🔄 Splitting query into {total_windows} time windows...")
                
                all_rows = []
                total_fetched = 0
                start_time_fetch = time.time()
                
                for window_num, (window_start, window_end) in enumerate(windows, 1):
                    log_db_operation(f"sequential_window_{window_num}", 
                                   params=f"Start: {window_start}, End: {window_end}",
                                   result=f"Starting window {window_num}/{total_windows}")
                    progress_container.text(f"🔄 Processing window {window_num}/{total_windows}: {window_start.strftime('%H:%M:%S')} to {window_end.strftime('%H:%M:%S')}")
                    
                    # Fetch window data
                    retries = 0
                    window_rows = None
                    last_error = None
                    conn = None
                    cursor = None
                    
                    while retries < MAX_RETRIES and window_rows is None:
                        try:
                            # Create direct connection (not from pool) to avoid stale connections
                            log_db_operation(f"sequential_window_{window_num}", result=f"Creating connection (attempt {retries + 1})")
                            conn = mysql.connector.connect(**DB_CONFIG)
                            if conn is None:
                                raise Exception("Failed to get database connection")
                            
                            log_db_operation(f"sequential_window_{window_num}", result="Connection created, setting timeouts")
                            # Set timeouts on this connection
                            setup_cursor = conn.cursor()
                            setup_cursor.execute("SET SESSION wait_timeout=28800")
                            setup_cursor.execute("SET SESSION interactive_timeout=28800")
                            setup_cursor.execute(f"SET SESSION net_read_timeout={LONG_NET_TIMEOUT}")
                            setup_cursor.execute(f"SET SESSION net_write_timeout={LONG_NET_TIMEOUT}")
                            setup_cursor.close()
                            
                            # Validate connection is alive
                            if not ping_connection(conn):
                                conn.close()
                                raise Exception("Connection is not alive")
                            
                            log_db_operation(f"sequential_window_{window_num}", result="Connection validated, building query")
                            cursor = conn.cursor(dictionary=True, buffered=False)
                            
                            query = f"SELECT {columns_str} FROM {TABLE_NAME} WHERE timestamp >= %s AND timestamp <= %s"
                            params = [window_start, window_end]
                            
                            if sensor_ids and len(sensor_ids) > 0:
                                placeholders = ','.join(['%s'] * len(sensor_ids))
                                query += f" AND sensor_id IN ({placeholders})"
                                params.extend(sensor_ids)
                            
                            if downsample and downsample > 1:
                                query += f" AND MOD(id, {downsample}) = 0"
                            
                            # Removed ORDER BY for speed - data is already roughly chronological by ID
                            # If ordering is needed, can sort in memory after fetching
                            query += f" LIMIT {MAX_ROWS_PER_WINDOW}"
                            
                            log_db_operation(f"sequential_window_{window_num}", query=query[:150], result="Executing query")
                            exec_start = time.time()
                            cursor.execute(query, params)
                            exec_time = time.time() - exec_start
                            log_db_operation(f"sequential_window_{window_num}", result=f"Query executed in {exec_time:.2f}s, starting fetch")
                            
                            window_rows = []
                            rows_fetched = 0
                            fetch_start = time.time()
                            while True:
                                fetch_chunk_start = time.time()
                                rows = cursor.fetchmany(CHUNK_SIZE)
                                fetch_chunk_time = time.time() - fetch_chunk_start
                                
                                if not rows:
                                    break
                                window_rows.extend(rows)
                                rows_fetched += len(rows)
                                
                                # Log progress every 10k rows
                                if rows_fetched % 10000 == 0:
                                    elapsed = time.time() - fetch_start
                                    log_db_operation(f"sequential_window_{window_num}", 
                                                    f"Progress: {rows_fetched:,} rows fetched in {elapsed:.2f}s")
                                
                                # Keep connection alive during long fetches - more frequent keepalive
                                if rows_fetched % 2000 == 0:  # More frequent keepalive
                                    try:
                                        # Use a lightweight keepalive - just execute a simple query
                                        keepalive_cursor = conn.cursor()
                                        keepalive_cursor.execute("SELECT 1")
                                        keepalive_cursor.fetchone()
                                        keepalive_cursor.close()
                                        log_db_operation(f"sequential_window_{window_num}", 
                                                        f"Keepalive ping successful at {rows_fetched:,} rows")
                                    except Exception as keepalive_err:
                                        # If keepalive fails, connection is lost
                                        error_msg = f"Connection lost during fetch at {rows_fetched:,} rows: {keepalive_err}"
                                        log_db_operation(f"sequential_window_{window_num}", error=error_msg)
                                        raise mysql.connector.Error(error_msg)
                            
                            total_fetch_time = time.time() - fetch_start
                            log_db_operation(f"sequential_window_{window_num}", 
                                           f"SUCCESS: Fetched {rows_fetched:,} rows in {total_fetch_time:.2f}s")
                            
                            cursor.close()
                            cursor = None
                            conn.close()
                            conn = None
                            
                        except mysql.connector.Error as db_err:
                            retries += 1
                            last_error = str(db_err)
                            error_code = db_err.errno if hasattr(db_err, 'errno') else None
                            log_db_operation(f"sequential_window_{window_num}", 
                                           error=f"Attempt {retries}/{MAX_RETRIES} failed: Error {error_code}: {last_error}")
                            logger.error(traceback.format_exc())
                            
                            # Clean up
                            if cursor:
                                try:
                                    cursor.close()
                                except:
                                    pass
                                cursor = None
                            if conn:
                                try:
                                    conn.close()
                                except:
                                    pass
                                conn = None
                            
                            # Clear connection pool cache if connection was lost
                            if error_code == 2013 or "Lost connection" in last_error:
                                # Force new connection on next retry
                                try:
                                    st.cache_resource.clear()
                                except:
                                    pass
                            
                            if retries < MAX_RETRIES:
                                # Longer delay for connection errors
                                if error_code == 2013 or "Lost connection" in last_error:
                                    delay = min(BASE_BACKOFF * (2 ** retries), MAX_BACKOFF * 2)  # Longer delay for connection issues
                                else:
                                    delay = min(BASE_BACKOFF * (2 ** (retries - 1)), MAX_BACKOFF)
                                
                                error_short = last_error[:100] if len(last_error) > 100 else last_error
                                log_db_operation(f"sequential_window_{window_num}", 
                                               f"Retrying in {delay}s (attempt {retries}/{MAX_RETRIES})")
                                if error_code == 2013 or "Lost connection" in last_error:
                                    progress_container.text(
                                        f"⚠️ Window {window_num}/{total_windows} - Connection lost. "
                                        f"Retrying with fresh connection in {delay}s (attempt {retries}/{MAX_RETRIES})..."
                                    )
                                else:
                                    progress_container.text(
                                        f"⚠️ Window {window_num}/{total_windows} failed: {error_short}. "
                                        f"Retrying in {delay}s (attempt {retries}/{MAX_RETRIES})..."
                                    )
                                time.sleep(delay)
                            else:
                                # Final attempt failed
                                error_short = last_error[:200] if len(last_error) > 200 else last_error
                                log_db_operation(f"sequential_window_{window_num}", 
                                               error=f"All {MAX_RETRIES} retries exhausted. Window failed: {error_short}")
                                return None, f"Failed to fetch window {window_num} after {MAX_RETRIES} attempts: {error_short}"
                        
                        except Exception as err:
                            retries += 1
                            last_error = str(err)
                            log_db_operation(f"sequential_window_{window_num}", 
                                           error=f"General Exception (attempt {retries}/{MAX_RETRIES}): {last_error}")
                            logger.error(traceback.format_exc())
                            
                            # Clean up
                            if cursor:
                                try:
                                    cursor.close()
                                except:
                                    pass
                                cursor = None
                            if conn:
                                try:
                                    conn.close()
                                except:
                                    pass
                                conn = None
                            
                            if retries < MAX_RETRIES:
                                delay = min(BASE_BACKOFF * (2 ** (retries - 1)), MAX_BACKOFF)
                                error_short = last_error[:100] if len(last_error) > 100 else last_error
                                progress_container.text(
                                    f"⚠️ Window {window_num}/{total_windows} failed: {error_short}. "
                                    f"Retrying in {delay}s (attempt {retries}/{MAX_RETRIES})..."
                                )
                                time.sleep(delay)
                            else:
                                # Final attempt failed
                                error_short = last_error[:200] if len(last_error) > 200 else last_error
                                return None, f"Failed to fetch window {window_num} after {MAX_RETRIES} attempts: {error_short}"
                
                if window_rows is not None:
                    all_rows.extend(window_rows)
                    total_fetched += len(window_rows)
                    log_db_operation(f"sequential_window_{window_num}", 
                                   f"Window completed successfully: {len(window_rows):,} rows added to results")
                    
                    progress = window_num / total_windows
                    progress_bar.progress(progress)
                    
                    elapsed = time.time() - start_time_fetch
                    if elapsed > 0:
                        rate = total_fetched / elapsed
                        progress_container.text(
                            f"✅ Window {window_num}/{total_windows} complete ({len(window_rows):,} rows) | "
                            f"Total: {total_fetched:,} rows | Rate: {rate:.0f} rows/sec"
                        )
                else:
                    error_msg = last_error if last_error else "Unknown error"
                    return None, f"Failed to fetch window {window_num}: {error_msg}"
        
        # Single query for small ranges
        else:
            status_text.text("🔄 Executing query...")
            progress_container.text("⏳ Fetching data...")
            
            # Use direct connection with extended timeouts
            conn = mysql.connector.connect(**DB_CONFIG)
            if conn is None:
                return None, "Failed to get database connection"
            
            # Set timeouts on this connection
            setup_cursor = conn.cursor()
            setup_cursor.execute("SET SESSION wait_timeout=28800")
            setup_cursor.execute("SET SESSION interactive_timeout=28800")
            setup_cursor.execute(f"SET SESSION net_read_timeout={LONG_NET_TIMEOUT}")
            setup_cursor.execute(f"SET SESSION net_write_timeout={LONG_NET_TIMEOUT}")
            setup_cursor.close()
            
            cursor = conn.cursor(dictionary=True, buffered=False)
            
            query = f"SELECT {columns_str} FROM {TABLE_NAME} WHERE timestamp >= %s AND timestamp <= %s"
            params = [start_time, end_time]
            
            if sensor_ids and len(sensor_ids) > 0:
                placeholders = ','.join(['%s'] * len(sensor_ids))
                query += f" AND sensor_id IN ({placeholders})"
                params.extend(sensor_ids)
            
            if downsample and downsample > 1:
                query += f" AND MOD(id, {downsample}) = 0"
            
            # Removed ORDER BY for speed - can sort in memory if needed
            # query += " ORDER BY timestamp ASC, id ASC"
            
            if limit_rows and limit_rows > 0:
                query += f" LIMIT {limit_rows}"
            
            cursor.execute(query, params)
            
            all_rows = []
            while True:
                rows = cursor.fetchmany(CHUNK_SIZE)
                if not rows:
                    break
                all_rows.extend(rows)
            
            cursor.close()
            conn.close()
        
        # Apply row limit if specified
        if limit_rows and limit_rows > 0 and len(all_rows) > limit_rows:
            all_rows = all_rows[:limit_rows]
        
        if not all_rows:
            progress_bar.empty()
            status_text.empty()
            progress_container.empty()
            return None, "No data found for the specified criteria."
        
        # Convert to DataFrame
        status_text.text("🔄 Converting to DataFrame...")
        df = pd.DataFrame(all_rows)
        del all_rows
        
        # Optimize data types
        status_text.text("🔄 Optimizing data types...")
        if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
        numeric_cols = ['b_x', 'b_y', 'b_z', 'lat', 'lon', 'alt', 'theta_x', 'theta_y', 'theta_z']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Successfully fetched {len(df):,} rows")
        progress_container.text(f"✅ Complete! Dataset size: {len(df):,} rows × {len(df.columns)} columns")
        
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        progress_container.empty()
        
        return df, f"Successfully fetched {len(df):,} rows"
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        progress_container.empty()
        return None, f"Error: {e}"


def main():
    logger.info("=" * 80)
    logger.info("🚀 Application started / Page reloaded")
    logger.info("=" * 80)
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Application started / Page reloaded")
    print(f"{'='*80}\n")
    
    st.title("🧲 Magnetic Data Fetcher")
    st.markdown("Fetch magnetic field data from the database for specific time periods")
    
    # Sidebar
    with st.sidebar:
        st.header("🔌 Database Connection")
        
        if st.button("Test Connection", type="primary"):
            log_user_action("Test Connection button clicked")
            success, message = test_connection()
            if success:
                log_user_action("Test Connection result", details="SUCCESS")
                st.success(message)
            else:
                log_user_action("Test Connection result", details=f"FAILED: {message}")
                st.error(message)
        
        st.divider()
        st.header("📊 Database Info")
        
        # Only fetch data range when button is clicked or if not cached
        if 'data_range' not in st.session_state:
            st.session_state.data_range = None
        
        if st.button("Check Data Range", use_container_width=True):
            log_user_action("Check Data Range button clicked")
            with st.spinner("Checking data range..."):
                min_date, max_date = get_data_range()
                log_user_action("Data range retrieved", details=f"MIN: {min_date}, MAX: {max_date}")
                st.session_state.data_range = (min_date, max_date)
        else:
            min_date, max_date = st.session_state.data_range if st.session_state.data_range else (None, None)
        
        if min_date and max_date:
            st.info(f"**Available Data Range:**\n\n"
                   f"Start: {min_date}\n\n"
                   f"End: {max_date}\n\n"
                   f"Total Duration: {(max_date - min_date).days} days")
        elif st.session_state.data_range is None:
            st.info("Click 'Check Data Range' to see available data")
        else:
            st.warning("Could not retrieve data range")
        
    # Main content
    st.header("📅 Select Time Period")
    
    # Quick presets
    st.subheader("⚡ Quick Presets")
    preset_cols = st.columns(5)
    
    presets = [
        ("Last Hour", timedelta(hours=1)),
        ("Last 24 Hours", timedelta(days=1)),
        ("Last Week", timedelta(days=7)),
        ("Last Month", timedelta(days=30)),
        ("Today", None)
    ]
    
    for idx, (label, delta) in enumerate(presets):
        with preset_cols[idx]:
            if st.button(label, use_container_width=True):
                if delta:
                    st.session_state.preset_start = datetime.now() - delta
                else:
                    today = datetime.now().date()
                    st.session_state.preset_start = datetime.combine(today, datetime.min.time())
                    st.session_state.preset_end = datetime.now()
                st.rerun()
    
    st.divider()
    
    # Time selection
    # Initialize session state for time values
    if 'start_date' not in st.session_state:
        st.session_state.start_date = (datetime.now() - timedelta(days=1)).date()
    if 'start_time' not in st.session_state:
        st.session_state.start_time = datetime.now().time().replace(second=0, microsecond=0)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now().date()
    if 'end_time' not in st.session_state:
        st.session_state.end_time = datetime.now().time().replace(second=0, microsecond=0)
    
    # Handle preset values
    if 'preset_start' in st.session_state:
        st.session_state.start_date = st.session_state.preset_start.date()
        st.session_state.start_time = st.session_state.preset_start.time()
        del st.session_state.preset_start
    if 'preset_end' in st.session_state:
        st.session_state.end_date = st.session_state.preset_end.date()
        st.session_state.end_time = st.session_state.preset_end.time()
        del st.session_state.preset_end
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.start_date,
            min_value=datetime(2025, 9, 26).date() if min_date is None else min_date.date(),
            max_value=datetime.now().date(),
            key="start_date_input"
        )
        start_time = st.time_input(
            "Start Time", 
            value=st.session_state.start_time,
            key="start_time_input"
        )
        # Update session state
        st.session_state.start_date = start_date
        st.session_state.start_time = start_time
        
        with col2:
            end_date = st.date_input(
                "End Date",
            value=st.session_state.end_date,
                min_value=datetime(2025, 9, 26).date() if min_date is None else min_date.date(),
                max_value=datetime.now().date(),
            key="end_date_input"
            )
            end_time = st.time_input(
                "End Time", 
            value=st.session_state.end_time,
            key="end_time_input"
            )
        # Update session state
        st.session_state.end_date = end_date
        st.session_state.end_time = end_time
        
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
    
    # Validate time range
    if end_datetime <= start_datetime:
        st.error("⚠️ End time must be after start time!")
        st.stop()
    
    duration = end_datetime - start_datetime
    st.info(f"📊 Selected time range: {duration.days} days, {duration.seconds // 3600} hours, {(duration.seconds % 3600) // 60} minutes")
    
    st.divider()
    
    # Column selection
    st.header("📋 Column Selection")
    all_columns = ["id", "sensor_id", "timestamp", "b_x", "b_y", "b_z", "lat", "lon", "alt", "theta_x", "theta_y", "theta_z"]
    selected_columns = st.multiselect(
        "Select columns to fetch:",
        options=all_columns,
        default=all_columns,
        help="Select which columns you want to include in the fetched data."
    )
    
    if not selected_columns:
            st.warning("⚠️ Please select at least one column!")
            st.stop()
    
    st.divider()
    
    # Filter options
    st.header("🔍 Filter Options")
    col1, col2 = st.columns(2)
    
    with col1:
        # Initialize sensors in session state if not present
        if 'available_sensors' not in st.session_state:
            st.session_state.available_sensors = None
        
        # Only load sensors when user clicks the button
        if st.button("📡 Load Sensors", type="primary", use_container_width=True):
            log_user_action("Load Sensors button clicked")
            with st.spinner("Loading sensors..."):
                sensors = get_available_sensors()
                st.session_state.available_sensors = sensors
                log_user_action("Sensors loaded", details=f"Count: {len(sensors) if sensors else 0}")
                if sensors and len(sensors) > 0:
                    st.success(f"✅ Loaded {len(sensors)} sensors")
                else:
                    st.warning("⚠️ No sensors found or failed to load")
        
        available_sensors = st.session_state.available_sensors
        
        if available_sensors and len(available_sensors) > 0:
            sensor_selection = st.multiselect(
                "Select Sensors (leave empty for all)",
                options=available_sensors,
                help="Select specific sensors or leave empty to fetch data from all sensors"
            )
            if sensor_selection:
                log_user_action("Sensors selected", details=f"Sensors: {sensor_selection}")
        else:
            if st.session_state.available_sensors is None:
                st.info("ℹ️ Click 'Load Sensors' to load available sensors. You can still fetch data without sensor filter.")
            else:
                st.warning("⚠️ Could not load sensor list. You can still fetch data without sensor filter.")
            sensor_selection = []
            
            # Add refresh button if sensors were already loaded
            if st.session_state.available_sensors is not None:
                if st.button("🔄 Retry Loading Sensors", use_container_width=True):
                    log_user_action("Retry Loading Sensors button clicked")
                    st.session_state.available_sensors = None
                    st.rerun()
    
    with col2:
        enable_downsample = st.checkbox("Enable Downsampling", value=False, 
                                        help="Reduce data size by keeping only every Nth row")
        if enable_downsample:
            downsample_factor = st.number_input(
                "Downsample Factor",
                min_value=2,
                max_value=1000,
                value=60,
                help="Keep only rows where MOD(id, factor) = 0"
            )
        else:
            downsample_factor = None
    
    st.divider()
    
    # Submit button
    st.header("🚀 Fetch Data")
    
    if st.button("Fetch Data", type="primary", use_container_width=True):
        log_user_action("Fetch Data button clicked", details=f"Start: {start_datetime}, End: {end_datetime}, Sensors: {sensor_selection}, Columns: {len(selected_columns)}, Downsample: {downsample_factor}")
        with st.spinner("Fetching data from database..."):
            df, message = fetch_data(
                start_datetime,
                end_datetime,
                columns=selected_columns,
                sensor_ids=sensor_selection if sensor_selection else None,
                downsample=downsample_factor
            )
            
            if df is not None:
                log_user_action("Fetch Data result", details=f"SUCCESS: {len(df)} rows fetched - {message}")
                st.session_state.fetched_data = df
                st.session_state.fetch_status = message
                st.success(message)
            else:
                log_user_action("Fetch Data result", details=f"FAILED: {message}")
                st.error(message)
                st.session_state.fetched_data = None
                st.session_state.fetch_status = None
    
    # Display fetched data
    if st.session_state.fetched_data is not None:
        df = st.session_state.fetched_data
        
        st.divider()
        st.header("📊 Fetched Data Preview")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Unique Sensors", df['sensor_id'].nunique())
        with col3:
            if df['timestamp'].notna().any():
                duration_actual = (df['timestamp'].max() - df['timestamp'].min())
                st.metric("Data Duration", f"{duration_actual.days}d {duration_actual.seconds//3600}h")
        with col4:
            st.metric("Columns", len(df.columns))
        
        # Data preview
        st.subheader("Data Preview (First 100 rows)")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Data statistics
        with st.expander("📈 Data Statistics"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Sensor breakdown
        if df['sensor_id'].nunique() > 1:
            with st.expander("🔍 Sensor Breakdown"):
                sensor_counts = df['sensor_id'].value_counts()
                try:
                    st.bar_chart(sensor_counts)
                except:
                    st.dataframe(sensor_counts.to_frame("Count"), use_container_width=True)
        
        st.divider()
        
        # Download section
        st.header("💾 Download Data")
        
        # Generate CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"magnetic_data_{start_datetime.strftime('%Y%m%d_%H%M%S')}_to_{end_datetime.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Clear data button
        if st.button("🗑️ Clear Fetched Data", use_container_width=True):
            st.session_state.fetched_data = None
            st.session_state.fetch_status = None
            st.rerun()


if __name__ == "__main__":
    main()
