"""Machine resources: worker threads and free memory."""

import ctypes
import os
import sys

MAX_WORKERS = 8   # beyond this, memory bandwidth limits the speed-up


def default_workers():
    """Worker threads to use: all cores but one (leaves the UI responsive), at most MAX_WORKERS."""
    return max(1, min(MAX_WORKERS, (os.cpu_count() or 2) - 1))


def available_memory():
    """Free physical memory in bytes, or None if it can't be determined."""
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
