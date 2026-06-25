"""
ROAD AI — GPS reader module (NEO-6M over UART)
Reads NMEA sentences in a background thread and exposes the latest fix.

Port is configurable via the GPS_PORT env var:
  - Jetson Nano 40-pin header (pins 8/10):  /dev/ttyTHS1  (default)
  - Raspberry Pi:                           /dev/ttyAMA0
Import and call start(), then get_location() for the latest (lat, lng).
"""

import os
import threading
import time

try:
    import serial
    import pynmea2
    _HAVE_SERIAL = True
except ImportError:
    _HAVE_SERIAL = False

GPS_PORT     = os.getenv("GPS_PORT", "/dev/ttyTHS1")
GPS_BAUDRATE = int(os.getenv("GPS_BAUDRATE", "9600"))

_lock    = threading.Lock()
_lat     = None
_lng     = None
_fix     = False
_running = False


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
                        if msg.status == "A":          # A = active fix
                            with _lock:
                                _lat = round(msg.latitude,  6)
                                _lng = round(msg.longitude, 6)
                                _fix = True
                        else:
                            with _lock:
                                _fix = False
        except Exception as e:
            print(f"[GPS] serial error: {e} — retrying in 3s")
            time.sleep(3)


def start():
    """Start the background NMEA reader. No-op (warns) if pyserial is missing."""
    global _running
    if not _HAVE_SERIAL:
        print("[GPS] pyserial/pynmea2 not installed — GPS disabled")
        return False
    if _running:
        return True
    _running = True
    threading.Thread(target=_reader_thread, daemon=True).start()
    print(f"[GPS] reading from {GPS_PORT} at {GPS_BAUDRATE} baud")
    return True


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
