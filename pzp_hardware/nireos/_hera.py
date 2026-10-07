# This file is a part of pzp-hardware, a library of laboratory hardware support Pieces
# for the puzzlepiece GUI & automation framework. Check out https://pzp-hardware.readthedocs.io
# Licensed under the Apache License 2.0 - https://github.com/jdranczewski/pzp-hardware/blob/main/LICENSE

"""
ctypes wrapper for the Nireos Hera C API (HeraAPI.dll, Hera SDK >= 1.3).

The Hera SDK treats a GEMINI-X interferometer and its camera (here a Basler a2A1280-125umSWIR)
as a single "HERA device": camera settings, live frames, and hyperspectral acquisitions
(interferometer scan + FT into a hypercube) all go through one device handle.

Header: C:\\Program Files\\Nireos\\Hera SDK\\HeraAPI\\include\\HeraAPI.h

A FakeHeraDevice with the same interface is provided for puzzlepiece debug mode.
"""
import atexit
import ctypes as C
import os
import threading
import time

import numpy as np

DEFAULT_DLL_DIR = r"C:\Program Files\Nireos\Hera SDK\HeraAPI\bin"

# HeraAPI.dll needs a recent MSVC C++ runtime (msvcp140.dll >= 14.40, the VS 2022 17.10 std::mutex
# change). PyQt5 bundles msvcp140.dll 14.26, and Windows reuses whichever msvcp140.dll a process
# loaded first, so if Qt is imported first HeraAPI_CreateDevice crashes with an access violation.
# Loading the system runtime before Qt avoids this. That is done here if this module is imported
# before Qt, otherwise by a .pth file in the Python environment (see the hyperspectral_camera docs).
MIN_MSVCP140_VERSION = (14, 40)
_SYSTEM_MSVCP140 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "msvcp140.dll")
if os.name == "nt":
    try:
        C.WinDLL(_SYSTEM_MSVCP140)
    except OSError:
        pass


def loaded_msvcp140():
    """Returns (path, version tuple) of the msvcp140.dll used by this process, or (None, None)."""
    if os.name != "nt":
        return None, None
    kernel32, version = C.WinDLL("kernel32"), C.WinDLL("version")
    kernel32.GetModuleHandleW.restype = C.c_void_p
    module = kernel32.GetModuleHandleW("msvcp140.dll")
    if not module:
        return None, None
    path = C.create_unicode_buffer(1024)
    kernel32.GetModuleFileNameW(C.c_void_p(module), path, 1024)
    size = version.GetFileVersionInfoSizeW(path.value, None)
    info = C.create_string_buffer(size)
    version.GetFileVersionInfoW(path.value, 0, size, info)
    fixed, length = C.c_void_p(), C.c_uint()
    version.VerQueryValueW(info, "\\", C.byref(fixed), C.byref(length))
    ms, ls = C.cast(fixed, C.POINTER(C.c_uint32 * 4)).contents[2:4]  # VS_FIXEDFILEINFO.dwFileVersionMS/LS
    return path.value, (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)


def runtime_problem():
    """Returns a description of the MSVC runtime problem above if this process has it, else None."""
    path, version = loaded_msvcp140()
    if version is None or version[:2] >= MIN_MSVCP140_VERSION:
        return None
    return (f"An old MSVC runtime ({path}, version {'.'.join(map(str, version))}) is loaded, which makes the "
            f"Hera SDK crash. Import pzp_hardware.nireos.hyperspectral_camera before PyQt5/puzzlepiece, or add "
            f"a .pth file to the Python environment that preloads {_SYSTEM_MSVCP140} at startup.")

# Enums from HeraAPI.h - list index == enum value
SCAN_MODES = ["Short", "Medium", "Long", "ExtraLong"]
PIXEL_FORMATS = ["Mono8", "Mono10", "Mono12", "Mono14", "Mono16"]
TRIGGER_MODES = ["Internal", "DeferredStartExtLineHi", "StepScanExtLoHi"]
TEMPERATURE_STATUS = {-1: "NotAvailable", 0: "Ok", 1: "Warning", 2: "Critical"}
DATA_TYPES = ["SinglePrecision", "DoublePrecision"]
BINNING = {"None": 0x0000, "Bin2x": 1, "Bin4x": 2, "Bin8x": 3, "Bin2xEnhanced": 0x1000, "Bin4xEnhanced": 0x1001}
RESOURCE_USAGE = ["LowUsage", "MediumUsage", "HighUsage", "VeryHighUsage", "MaxUsage"]
DATA_STATUS = ["OK", "ABORTED", "DEVICE_TIMEOUT", "DEVICE_LOST", "CAMERA_INIT_ERROR", "CAPTURE_ERROR", "UNKNOWN"]

STATUS_NAMES = {
    0x00000000: "HERA_API_OK",
    0x80000008: "HERA_API_FAIL",
    0x80000009: "HERA_API_CORRUPTED",
    0x8000000A: "HERA_API_NOT_LICENSED",
    0x8000000B: "HERA_API_CANNOT_APPLY_LICENSE",
    0x8000000C: "HERA_API_INVALID_ARG",
    0x8000000D: "HERA_API_INVALID_ARG_OUTOFBOUNDS",
    0x8000000E: "HERA_API_INVALID_HANDLE",
    0x8000000F: "HERA_API_DEVICE_ALREADY_CREATED",
    0x80000010: "HERA_API_CANNOT_SET_PARAMETER",
    0x80000011: "HERA_API_CANNOT_GET_PARAMETER",
    0x80000012: "HERA_API_CANNOT_EXECUTE_FUNCTION",
}


class HeraError(Exception):
    def __init__(self, function, status, message):
        self.status = status
        name = STATUS_NAMES.get(status, f"0x{status:08X}")
        super().__init__(f"{function} failed with {name}: {message}")


class HeraDeviceInfo(C.Structure):
    # The public header adds a `HeraDeviceInfoEx` macro at the end of the struct which expands
    # to nothing for us, but may hold private fields in the DLL's own build. Pad generously so
    # the DLL can never write past our allocation.
    _fields_ = [
        ("Id", C.c_char * 128),
        ("ProductName", C.c_char * 128),
        ("SerialNumber", C.c_char * 128),
        ("Vendor", C.c_char * 128),
        ("_reserved", C.c_char * 4096),
    ]

    def as_dict(self):
        return {k: getattr(self, k).decode(errors="replace") for k in ("Id", "ProductName", "SerialNumber", "Vendor")}


# Callback types (HERA_API_CC is __cdecl)
VoidCallback = C.CFUNCTYPE(None)
StrCallback = C.CFUNCTYPE(None, C.c_char_p)
IntCallback = C.CFUNCTYPE(None, C.c_int)
FloatCallback = C.CFUNCTYPE(None, C.c_float)
LiveCaptureCallback = C.CFUNCTYPE(None, C.c_void_p)
HyperspectralDataCallback = C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_char_p)

_H = C.c_void_p
_P = C.POINTER
_SIGNATURES = {
    "GetVersion": [_P(C.c_int)] * 3,
    "SetResourceUsage": [C.c_int],
    "Cleanup": [],
    "IsLicensed": [_P(C.c_bool), _P(C.c_longlong), _P(C.c_longlong)],
    "ActivateLicenseOnline": [C.c_char_p],
    "ActivateLicenseOffline": [C.c_char_p, C.c_char_p],
    "EnumerateDevices": [_P(C.c_size_t)],
    "GetDeviceInfoByIndex": [C.c_size_t, _P(HeraDeviceInfo)],
    "GetDeviceInfo": [_H, _P(HeraDeviceInfo)],
    "CreateDevice": [_P(HeraDeviceInfo), _P(_H)],
    "ReleaseDevice": [_H],
    "Connect": [_H],
    "Disconnect": [_H],
    "IsConnected": [_H, _P(C.c_bool)],
    "RegisterDeviceLostCallback": [_H, VoidCallback],
    "UnregisterDeviceLostCallback": [_H],
    "RegisterLiveCaptureCallbacks": [_H, StrCallback, IntCallback, LiveCaptureCallback],
    "UnregisterLiveCaptureCallbacks": [_H],
    "RegisterHyperspectralDataAcqCallbacks": [_H, FloatCallback, HyperspectralDataCallback],
    "UnregisterHyperspectralDataAcqCallbacks": [_H],
    "GetGainLevelResolution": [_H, _P(C.c_double)],
    "GetGainLevel": [_H, _P(C.c_double)],
    "SetGainLevel": [_H, C.c_double],
    "GetActualGain": [_H, _P(C.c_double)],
    "IsGainLevelWritable": [_H, _P(C.c_bool)],
    "SetExposure": [_H, C.c_double],
    "GetExposure": [_H, _P(C.c_double)],
    "IsExposureWritable": [_H, _P(C.c_bool)],
    "IsBlackLevelSetSupported": [_H, _P(C.c_bool)],
    "SetBlackLevel": [_H, C.c_double],
    "GetBlackLevel": [_H, _P(C.c_double)],
    "IsBlackLevelWritable": [_H, _P(C.c_bool)],
    "SetROI": [_H, C.c_uint, C.c_uint, C.c_uint, C.c_uint],
    "ClearROI": [_H],
    "GetOffsetX": [_H, _P(C.c_uint)],
    "GetOffsetY": [_H, _P(C.c_uint)],
    "GetWidth": [_H, _P(C.c_uint)],
    "GetHeight": [_H, _P(C.c_uint)],
    "IsROIWritable": [_H, _P(C.c_bool)],
    "GetTemperatureInfo": [_H, _P(C.c_int), _P(C.c_double)],
    "IsPixelFormatSupported": [_H, C.c_int, _P(C.c_bool)],
    "IsPixelFormatSupportedEx": [_H, C.c_int, C.c_bool, _P(C.c_bool)],
    "IsHDRSupported": [_H, _P(C.c_bool)],
    "SetHDR": [_H, C.c_bool],
    "GetHDR": [_H, _P(C.c_bool)],
    "StartLiveCapture": [_H, C.c_int],
    "StopLiveCapture": [_H],
    "IsLiveCapturing": [_H, _P(C.c_bool)],
    "IsTriggerModeSupported": [_H, C.c_int, _P(C.c_bool)],
    "IsExternalTriggerDelaySupported": [_H, _P(C.c_bool)],
    "SetExternalTriggerDelay": [_H, C.c_double],
    "GetExternalTriggerDelay": [_H, _P(C.c_double)],
    "IsScanModeSupported": [_H, C.c_int, _P(C.c_bool)],
    "GetSupportedScanModes": [_H, _P(C.c_int * 16), _P(C.c_size_t)],
    "GetDefaultOutBands": [_H, C.c_int, _P(C.c_int)],
    "StartHyperspectralDataAcquisitionEx": [_H, C.c_int, C.c_int, C.c_int, C.c_int],
    "AbortHyperspectralDataAcquisition": [_H],
    "IsAcquiringHyperspectralData": [_H, _P(C.c_bool)],
    "GetHyperCubeEx": [_H, _H, _P(_H), C.c_int, C.c_uint, C.c_int, C.c_void_p],
    "GetHyperCubeInfo": [_H, _P(C.c_int), _P(C.c_int), _P(C.c_int), _P(C.c_int)],
    "GetHyperCubeBandData": [_H, C.c_uint, _P(C.c_double), _P(C.c_void_p)],
    "ExportHyperCubeAsEnvi": [_H, C.c_char_p, C.c_char_p],
    "ReleaseHyperCube": [_H],
    "GetLiveCaptureInfo": [_H] + [_P(C.c_int)] * 6 + [_P(C.c_void_p)],
    "GetLiveCaptureIsHDR": [_H, _P(C.c_bool)],
    "ReleaseLiveCaptureResult": [_H],
    "ExportHyperspectralData": [_H, C.c_char_p, C.c_bool],
    "ReleaseHyperspectralData": [_H],
}


class HeraSDK:
    """Loads HeraAPI.dll and exposes the library-level (not device-level) functions."""

    def __init__(self, dll_dir=DEFAULT_DLL_DIR):
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]
        os.add_dll_directory(dll_dir)
        self.dll = C.CDLL(os.path.join(dll_dir, "HeraAPI.dll"))
        self.dll.HeraAPI_GetLastErrorMessage.restype = C.c_char_p
        self.dll.HeraAPI_GetLastErrorMessage.argtypes = []
        for name, argtypes in _SIGNATURES.items():
            f = getattr(self.dll, "HeraAPI_" + name)
            f.argtypes = argtypes
            f.restype = C.c_uint32
        # Open HeraDevices. If Python exits without closing them (e.g. the script raised before the
        # Puzzle was closed) they are closed at exit - with live capture still running, the process
        # otherwise hangs on exit inside the camera driver.
        self.devices = set()
        self._cleaned_up = False
        atexit.register(self._close_at_exit)

    def call(self, name, *args):
        status = getattr(self.dll, "HeraAPI_" + name)(*args)
        if status != 0:
            message = self.dll.HeraAPI_GetLastErrorMessage()
            raise HeraError("HeraAPI_" + name, status, message.decode(errors="replace") if message else "")

    @property
    def version(self):
        v = [C.c_int() for _ in range(3)]
        self.call("GetVersion", *[C.byref(x) for x in v])
        return ".".join(str(x.value) for x in v)

    def license_status(self):
        """Returns (is_licensed, licence_expiry_utc, certificate_expiry_utc) - expiries are unix timestamps or 0."""
        lic, lexp, cexp = C.c_bool(), C.c_longlong(), C.c_longlong()
        self.call("IsLicensed", C.byref(lic), C.byref(lexp), C.byref(cexp))
        return lic.value, lexp.value, cexp.value

    def activate_license_online(self, key):
        self.call("ActivateLicenseOnline", key.encode())

    def activate_license_offline(self, key, certificate):
        self.call("ActivateLicenseOffline", key.encode(), certificate.encode())

    def set_resource_usage(self, usage):
        self.call("SetResourceUsage", RESOURCE_USAGE.index(usage))

    def discover(self):
        """Returns a list of HeraDeviceInfo structs for all connected devices."""
        n = C.c_size_t()
        self.call("EnumerateDevices", C.byref(n))
        infos = []
        for i in range(n.value):
            info = HeraDeviceInfo()
            self.call("GetDeviceInfoByIndex", i, C.byref(info))
            infos.append(info)
        return infos

    def discover_serials(self):
        return [info.SerialNumber.decode(errors="replace") for info in self.discover()]

    def open_device(self, serial=None):
        problem = runtime_problem()
        if problem:
            raise Exception(problem)
        infos = self.discover()
        if not infos:
            raise Exception("No HERA devices found")
        if serial:
            infos = [i for i in infos if i.SerialNumber.decode(errors="replace") == serial]
            if not infos:
                raise Exception(f"HERA device '{serial}' not found")
        return HeraDevice(self, infos[0])

    def cleanup(self):
        if not self._cleaned_up:
            self._cleaned_up = True
            self.call("Cleanup")

    def _close_at_exit(self):
        for device in list(self.devices):
            try:
                device.close()
            except Exception as e:
                print(f"hera: error closing {device.serial} at exit: {e}")
        try:
            self.cleanup()
        except Exception as e:
            print(f"hera: error in SDK cleanup at exit: {e}")


class _FrameBuffer:
    """Thread-safe store for the latest frame, with a counter so callers can wait for a new one."""

    def __init__(self):
        self.cond = threading.Condition()
        self.frame = None
        self.count = 0
        self.error = None

    def put(self, frame):
        with self.cond:
            self.frame = frame
            self.count += 1
            self.cond.notify_all()

    def put_error(self, message):
        with self.cond:
            self.error = message
            self.cond.notify_all()

    def wait(self, min_count, timeout):
        """Wait until at least `min_count` frames have arrived, return (frame, count)."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while self.count < min_count:
                if self.error is not None:
                    error, self.error = self.error, None
                    raise Exception(f"Live capture error: {error}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Acquisition did not complete within the timeout...")
                self.cond.wait(remaining)
            return self.frame, self.count


class HeraDevice:
    """A connected HERA device (interferometer + camera)."""

    def __init__(self, sdk, info):
        self.sdk = sdk
        self.info = info.as_dict()
        self.handle = C.c_void_p()
        self.frames = _FrameBuffer()
        self.last_cube = None  # HyperCubeHandle of the latest acquisition, kept for ENVI export
        # HyperspectralDataHandle (raw interferograms) of the latest acquisition, kept so the cube
        # can be recomputed with other settings without a new scan
        self._data_handle = None
        self._data_scan_mode = None
        sdk.call("CreateDevice", C.byref(info), C.byref(self.handle))
        try:
            if not self.is_connected:
                sdk.call("Connect", self.handle)
            self._register_callbacks()
        except Exception:
            sdk.call("ReleaseDevice", self.handle)
            raise
        sdk.devices.add(self)

    # region helpers
    def _call(self, name, *args):
        self.sdk.call(name, self.handle, *args)

    def _get(self, name, ctype, *args):
        value = ctype()
        self._call(name, *args, C.byref(value))
        return value.value

    # endregion

    # region connection
    @property
    def serial(self):
        return self.info["SerialNumber"]

    @property
    def is_connected(self):
        return self._get("IsConnected", C.c_bool)

    def close(self):
        """Stop everything, disconnect, and release the device handle. Safe to call twice."""
        if self.handle is None:
            return
        try:
            if self.is_acquiring_cube:
                self.abort_cube()
            if self.is_live:
                self.stop_live()
            self._release_cube()
            self._release_data()
            for name in ("UnregisterLiveCaptureCallbacks", "UnregisterHyperspectralDataAcqCallbacks",
                         "UnregisterDeviceLostCallback"):
                try:
                    self._call(name)
                except HeraError:
                    pass
            self._call("Disconnect")
        finally:
            try:
                self.sdk.call("ReleaseDevice", self.handle)
            finally:
                self.handle = None
                self.sdk.devices.discard(self)

    def _register_callbacks(self):
        # References are kept on self so the CFUNCTYPE objects are not garbage collected
        self._cb_live_error = StrCallback(lambda msg: self.frames.put_error(msg.decode(errors="replace") if msg else "unknown"))
        self._cb_live_timeout = IntCallback(lambda t: self.frames.put_error(f"timeout ({t})"))
        self._cb_live_frame = LiveCaptureCallback(self._on_live_frame)
        self._call("RegisterLiveCaptureCallbacks", self._cb_live_error, self._cb_live_timeout, self._cb_live_frame)

        self._cube_event = threading.Event()
        self._cube_progress = 0.0
        self._cube_result = None
        self._cb_cube_progress = FloatCallback(self._on_cube_progress)
        self._cb_cube_data = HyperspectralDataCallback(self._on_cube_data)
        self._call("RegisterHyperspectralDataAcqCallbacks", self._cb_cube_progress, self._cb_cube_data)

        self.lost = False
        self._cb_lost = VoidCallback(self._on_lost)
        self._call("RegisterDeviceLostCallback", self._cb_lost)

    def _on_lost(self):
        self.lost = True
        self.frames.put_error("device lost")

    # endregion

    # region camera settings
    @property
    def exposure_us(self):
        return self._get("GetExposure", C.c_double)

    @exposure_us.setter
    def exposure_us(self, value):
        self._call("SetExposure", float(value))

    @property
    def gain_level(self):
        return self._get("GetGainLevel", C.c_double)

    @gain_level.setter
    def gain_level(self, value):
        # 0..1, mapped by the camera onto its full analogue gain range (0..42 dB on the a2A1280)
        self._call("SetGainLevel", float(value))

    @property
    def gain_level_resolution(self):
        return self._get("GetGainLevelResolution", C.c_double)

    @property
    def actual_gain(self):
        return self._get("GetActualGain", C.c_double)

    @property
    def black_level_supported(self):
        return self._get("IsBlackLevelSetSupported", C.c_bool)

    @property
    def black_level(self):
        return self._get("GetBlackLevel", C.c_double)

    @black_level.setter
    def black_level(self, value):
        self._call("SetBlackLevel", float(value))

    @property
    def roi(self):
        """(offset_x, offset_y, width, height)"""
        return tuple(self._get(n, C.c_uint) for n in ("GetOffsetX", "GetOffsetY", "GetWidth", "GetHeight"))

    @roi.setter
    def roi(self, value):
        x, y, w, h = (int(v) for v in value)
        self._call("SetROI", x, y, w, h)

    def clear_roi(self):
        self._call("ClearROI")

    @property
    def temperature(self):
        """(status string, temperature in C)"""
        status, temp = C.c_int(), C.c_double()
        self._call("GetTemperatureInfo", C.byref(status), C.byref(temp))
        return TEMPERATURE_STATUS.get(status.value, str(status.value)), temp.value

    def pixel_format_supported(self, pixel_format, hdr=False):
        return self._get("IsPixelFormatSupportedEx", C.c_bool, PIXEL_FORMATS.index(pixel_format), bool(hdr))

    @property
    def hdr_supported(self):
        return self._get("IsHDRSupported", C.c_bool)

    @property
    def hdr(self):
        return self._get("GetHDR", C.c_bool)

    @hdr.setter
    def hdr(self, value):
        self._call("SetHDR", bool(value))

    # endregion

    # region live capture
    def start_live(self, pixel_format="Mono12"):
        self._call("StartLiveCapture", PIXEL_FORMATS.index(pixel_format))

    def stop_live(self):
        self._call("StopLiveCapture")

    @property
    def is_live(self):
        return self._get("IsLiveCapturing", C.c_bool)

    def _on_live_frame(self, capture_handle):
        # Runs on the camera thread: copy the frame out and release the SDK buffer immediately
        try:
            w, h, depth, bpp, sat, stride = (C.c_int() for _ in range(6))
            data = C.c_void_p()
            self.sdk.call("GetLiveCaptureInfo", capture_handle, *(C.byref(v) for v in (w, h, depth, bpp, sat, stride)),
                          C.byref(data))
            hdr = C.c_bool()
            try:
                self.sdk.call("GetLiveCaptureIsHDR", capture_handle, C.byref(hdr))
            except HeraError:
                pass
            dtype = {8: np.uint8, 16: np.uint16}.get(bpp.value, np.float32 if hdr.value else np.uint32)
            itemsize = np.dtype(dtype).itemsize
            # rowStride is in bytes (verified: width 400, Mono12 -> stride 800), the pixel fallback is just a guard
            stride_bytes = stride.value if stride.value >= w.value * itemsize else stride.value * itemsize
            buffer = (C.c_ubyte * (stride_bytes * h.value)).from_address(data.value)
            frame = np.frombuffer(buffer, dtype=dtype).reshape(h.value, stride_bytes // itemsize)[:, :w.value].copy()
            self.frames.put(frame)
        except Exception as e:
            self.frames.put_error(str(e))
        finally:
            try:
                self.sdk.call("ReleaseLiveCaptureResult", capture_handle)
            except HeraError:
                pass

    def get_frame(self, fresh=True, timeout=5.0):
        """
        Return the latest live frame. If `fresh`, wait for a frame whose exposure started after this
        call (we skip the frame that may already be exposing), otherwise just wait for any frame
        that has not been returned yet.
        """
        count = self.frames.count
        frame, _ = self.frames.wait(count + (2 if fresh else 1), timeout)
        return frame

    # endregion

    # region hyperspectral
    @property
    def supported_scan_modes(self):
        modes, n = (C.c_int * 16)(), C.c_size_t()
        self._call("GetSupportedScanModes", C.byref(modes), C.byref(n))
        return [SCAN_MODES[modes[i]] for i in range(n.value)]

    def default_out_bands(self, scan_mode):
        return self._get("GetDefaultOutBands", C.c_int, SCAN_MODES.index(scan_mode))

    def trigger_mode_supported(self, trigger_mode):
        return self._get("IsTriggerModeSupported", C.c_bool, TRIGGER_MODES.index(trigger_mode))

    @property
    def external_trigger_delay_supported(self):
        return self._get("IsExternalTriggerDelaySupported", C.c_bool)

    @property
    def external_trigger_delay_us(self):
        return self._get("GetExternalTriggerDelay", C.c_double)

    @external_trigger_delay_us.setter
    def external_trigger_delay_us(self, value):
        self._call("SetExternalTriggerDelay", float(value))

    @property
    def is_acquiring_cube(self):
        return self._get("IsAcquiringHyperspectralData", C.c_bool)

    def _on_cube_progress(self, progress):
        self._cube_progress = progress

    def _on_cube_data(self, data_handle, status, message):
        self._cube_result = (data_handle, status, message.decode(errors="replace") if message else "")
        self._cube_event.set()

    def start_cube(self, scan_mode="Short", trigger_mode="Internal", averages=1, stabilization_ms=0):
        self._release_data()
        self._cube_event.clear()
        self._cube_result = None
        self._cube_progress = 0.0
        self._scan_mode = scan_mode
        self._call("StartHyperspectralDataAcquisitionEx", SCAN_MODES.index(scan_mode),
                   TRIGGER_MODES.index(trigger_mode), int(averages), int(stabilization_ms))

    @property
    def cube_progress(self):
        return self._cube_progress

    @property
    def cube_done(self):
        return self._cube_event.is_set()

    def abort_cube(self):
        self._call("AbortHyperspectralDataAcquisition")

    def _release_cube(self):
        if self.last_cube is not None:
            try:
                self.sdk.call("ReleaseHyperCube", self.last_cube)
            except HeraError:
                pass
            self.last_cube = None

    def _release_data(self):
        if self._data_handle:
            try:
                self.sdk.call("ReleaseHyperspectralData", self._data_handle)
            except HeraError:
                pass
        self._data_handle = None

    @property
    def has_raw_data(self):
        """True if there is raw data from a finished acquisition that recompute_cube can use."""
        return bool(self._data_handle)

    def finish_cube(self, bands=0, binning="None", double=False, raw_path=None):
        """
        After cube_done, compute the hypercube from the raw interferogram data.
        Returns (wavelengths [nm], cube with shape (bands, height, width)), sorted by wavelength.
        `bands=0` uses the SDK default for the scan mode (the SDK itself rejects 0). `binning` bins
        the image spatially (Bin2x halves width and height). If `raw_path` is given, the raw data
        is also exported there (HeraAPI_ExportHyperspectralData).
        The raw data is kept until the next acquisition, see recompute_cube.
        """
        data_handle, status, message = self._cube_result
        self._cube_result = None
        if status != 0:
            if data_handle:
                self.sdk.call("ReleaseHyperspectralData", data_handle)
            raise Exception(f"Hyperspectral acquisition failed: {DATA_STATUS[status] if status < len(DATA_STATUS) else status} {message}")
        self._release_data()
        self._data_handle = data_handle
        self._data_scan_mode = self._scan_mode
        if raw_path:
            self.sdk.call("ExportHyperspectralData", data_handle, raw_path.encode(), True)
        return self.recompute_cube(bands, binning, double)

    def recompute_cube(self, bands=0, binning="None", double=False):
        """
        Compute the hypercube again from the raw data of the latest acquisition, e.g. with a
        different number of bands or binning (like "Recompute hypercube" in the HERA app).
        Returns (wavelengths [nm], cube) like finish_cube.
        """
        if not self._data_handle:
            raise Exception("No hyperspectral data to compute a cube from, acquire a cube first")
        if bands <= 0:
            bands = self.default_out_bands(self._data_scan_mode)
        self._release_cube()
        cube_handle = C.c_void_p()
        self.sdk.call("GetHyperCubeEx", self.handle, self._data_handle, C.byref(cube_handle), int(double),
                      int(bands), BINNING[binning], None)
        self.last_cube = cube_handle

        w, h, n, dtype = (C.c_int() for _ in range(4))
        self.sdk.call("GetHyperCubeInfo", cube_handle, C.byref(w), C.byref(h), C.byref(n), C.byref(dtype))
        np_dtype = np.float64 if dtype.value == 1 else np.float32
        ctype = C.c_double if dtype.value == 1 else C.c_float
        cube = np.empty((n.value, h.value, w.value), dtype=np_dtype)
        wls = np.empty(n.value)
        wl, data = C.c_double(), C.c_void_p()
        for i in range(n.value):
            self.sdk.call("GetHyperCubeBandData", cube_handle, i, C.byref(wl), C.byref(data))
            wls[i] = wl.value
            buffer = (ctype * (w.value * h.value)).from_address(data.value)
            cube[i] = np.frombuffer(buffer, dtype=np_dtype).reshape(h.value, w.value)
        order = np.argsort(wls)
        return wls[order], cube[order]

    def export_envi(self, path, description=""):
        if self.last_cube is None:
            raise Exception("No hypercube acquired yet")
        self.sdk.call("ExportHyperCubeAsEnvi", self.last_cube, path.encode(), description.encode())

    # endregion


class FakeHeraDevice:
    """Simulated HERA device with the same interface as HeraDevice, used in debug mode."""

    sensor = (1280, 1024)
    wl_range = (400.0, 1700.0)
    _default_bands = 120  # the real GVS2602006 returns 120 for every scan mode

    def __init__(self, serial="DEBUG0001"):
        self.info = {"Id": "debug", "ProductName": "Hera VIS-SWIR (simulated)", "SerialNumber": serial, "Vendor": "Nireos"}
        self.exposure_us = 25000.0
        self._gain_level = 0.0
        self.gain_level_resolution = 1 / 420
        self.black_level = 0.0
        self.black_level_supported = True
        self.hdr_supported = False
        self.hdr = False
        self.external_trigger_delay_supported = True
        self.external_trigger_delay_us = 0.0
        self.supported_scan_modes = ["Short", "Medium", "Long"]
        self.is_connected = True
        self.is_live = False
        self.is_acquiring_cube = False
        self.cube_progress = 0.0
        self.last_cube = None
        self._roi = (0, 0) + self.sensor
        self._pixel_format = "Mono12"
        self._cube_start = None
        self._raw = None
        self._cube_duration = 0.5
        self._frame_count = 0
        self._rng = np.random.default_rng()

    serial = property(lambda self: self.info["SerialNumber"])
    actual_gain = property(lambda self: self.gain_level * 42)
    temperature = property(lambda self: ("NotAvailable", 47.0 + self._rng.normal(0, 0.1)))

    @property
    def gain_level(self):
        return self._gain_level

    @gain_level.setter
    def gain_level(self, value):
        if not 0 <= value <= 1:
            raise HeraError("HeraAPI_SetGainLevel", 0x80000008, "gainLevel must be in the [0, 1] range")
        self._gain_level = float(value)

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, value):
        x, y, w, h = (int(v) for v in value)
        if x < 0 or y < 0 or w < 1 or h < 1 or x + w > self.sensor[0] or y + h > self.sensor[1]:
            raise HeraError("HeraAPI_SetROI", 0x8000000D, "ROI out of bounds")
        # Like the real camera, the x offset and width are rounded up to multiples of 4
        x, w = -(-x // 4) * 4, -(-w // 4) * 4
        self._roi = (x, y, min(w, self.sensor[0] - x), h)

    def clear_roi(self):
        self._roi = (0, 0) + self.sensor

    def close(self):
        self.is_live = False
        self.is_connected = False

    def pixel_format_supported(self, pixel_format, hdr=False):
        return not hdr

    def trigger_mode_supported(self, trigger_mode):
        return True

    def default_out_bands(self, scan_mode):
        if scan_mode not in self.supported_scan_modes:
            raise HeraError("HeraAPI_GetDefaultOutBands", 0x8000000C, "scan mode not supported")
        return self._default_bands

    def start_live(self, pixel_format="Mono12"):
        self._pixel_format = pixel_format
        self.is_live = True

    def stop_live(self):
        self.is_live = False

    def _scene(self):
        """A smooth spot on the sensor, cropped to the ROI, normalised to 0..1."""
        x0, y0, w, h = self._roi
        yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
        cx, cy = self.sensor[0] / 2, self.sensor[1] / 2
        return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 150.0 ** 2))

    def get_frame(self, fresh=True, timeout=5.0):
        if not self.is_live:
            raise Exception("Live capture error: not capturing")
        time.sleep(min(self.exposure_us * 1e-6, 0.05))
        self._frame_count += 1
        bits = int(self._pixel_format[4:])
        full = 2 ** bits - 1
        signal = self._scene() * full * 0.3 * (self.exposure_us / 25000.0) * 10 ** (self.actual_gain / 20)
        noise = self._rng.normal(0, 0.01 * full, signal.shape)
        frame = np.clip(signal + noise + self.black_level, 0, full)
        return frame.astype(np.uint8 if bits == 8 else np.uint16)

    def start_cube(self, scan_mode="Short", trigger_mode="Internal", averages=1, stabilization_ms=0):
        if scan_mode not in self.supported_scan_modes:
            raise HeraError("HeraAPI_StartHyperspectralDataAcquisitionEx", 0x8000000C, "scan mode not supported")
        self.is_live = False  # like the real device, a scan stops live capture
        self._raw = None
        self._scan_mode = scan_mode
        self._cube_duration = 0.2 * (1 + SCAN_MODES.index(scan_mode)) * averages + stabilization_ms / 1000
        self._cube_start = time.monotonic()
        self.is_acquiring_cube = True

    @property
    def cube_done(self):
        if self._cube_start is None:
            return False
        elapsed = time.monotonic() - self._cube_start
        self.cube_progress = min(elapsed / self._cube_duration, 1.0)
        done = elapsed >= self._cube_duration
        if done:
            self.is_acquiring_cube = False
        return done

    def abort_cube(self):
        self._cube_start = None
        self.is_acquiring_cube = False

    @property
    def has_raw_data(self):
        return self._raw is not None

    def finish_cube(self, bands=0, binning="None", double=False, raw_path=None):
        # "Raw data": what the scene looked like during the scan
        self._raw = (self._scene(), self.exposure_us)
        self._cube_start = None
        return self.recompute_cube(bands, binning, double)

    def recompute_cube(self, bands=0, binning="None", double=False):
        if self._raw is None:
            raise Exception("No hyperspectral data to compute a cube from, acquire a cube first")
        scene, exposure_us = self._raw
        n = bands if bands > 0 else self._default_bands
        wls = np.linspace(*self.wl_range, n)
        # Two emission lines on a broad background; the lines move across the sensor in x
        b = {"None": 1, "Bin2x": 2, "Bin4x": 4, "Bin8x": 8, "Bin2xEnhanced": 2, "Bin4xEnhanced": 4}[binning]
        h, w = scene.shape[0] // b, scene.shape[1] // b
        scene = scene[:h * b, :w * b].reshape(h, b, w, b).mean(axis=(1, 3))
        x = np.linspace(0, 1, scene.shape[1])
        centre = 900 + 300 * x
        spectrum = (np.exp(-(wls[:, None, None] - centre[None, None, :]) ** 2 / (2 * 15 ** 2))
                    + 0.5 * np.exp(-(wls[:, None, None] - 1300) ** 2 / (2 * 40 ** 2))
                    + 0.05)
        cube = scene[None] * spectrum * (exposure_us / 25000.0)
        cube = cube + self._rng.normal(0, 0.005, cube.shape)
        self.last_cube = (wls, cube)
        return wls, cube.astype(np.float64 if double else np.float32)

    def export_envi(self, path, description=""):
        if self.last_cube is None:
            raise Exception("No hypercube acquired yet")
        wls, cube = self.last_cube
        np.savez(path, wls=wls, cube=cube, description=description)
