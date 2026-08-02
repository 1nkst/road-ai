"""
ROAD AI — GPS reader module for Raspberry Pi
Reads NMEA sentences from NEO-6M via /dev/ttyAMA0 in a background thread.
Import and call get_location() to get the latest fix.
"""

import threading
import serial
import pynmea2
import time

GPS_PORT     = "/dev/ttyAMA0"
GPS_BAUDRATE = 9600

_lock     = threading.Lock()
_lat      = None
_lng      = None
_fix      = False
_running  = False


def _reader_thread():
    global _lat, _lng, _fix
    while _running:
        try:
            with serial.Serial(GPS_PORT, GPS_BAUDRATE, timeout=1) as ser:
                while _running:
                    line = ser.readline().decode("ascii", errors="replace").strip()
                    if not line.startswith("$GPRMC") and not line.startswith("$GPGGA"):
                        continue
                    try:
                        msg = pynmea2.parse(line)
                    except pynmea2.ParseError:
                        continue

                    if isinstance(msg, pynmea2.types.talker.RMC):
                        if msg.status == "A":   # A = active fix
                            with _lock:
                                _lat = round(msg.latitude,  6)
                                _lng = round(msg.longitude, 6)
                                _fix = True
                        else:
                            with _lock:
                                _fix = False

        except serial.SerialException as e:
            print(f"[GPS] serial error: {e} — retrying in 3s")
            time.sleep(3)


def start():
    global _running
    _running = True
    t = threading.Thread(target=_reader_thread, daemon=True)
    t.start()
    print(f"[GPS] reading from {GPS_PORT} at {GPS_BAUDRATE} baud")


def stop():
    global _running
    _running = False


def get_location():
    """Return (lat, lng) if a fix is available, else (None, None)."""
    with _lock:
        if _fix:
            return _lat, _lng
        return None, None


def has_fix():
    with _lock:
        return _fix
