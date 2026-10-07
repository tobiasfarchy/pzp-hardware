# This file is a part of pzp-hardware, a library of laboratory hardware support Pieces
# for the puzzlepiece GUI & automation framework. Check out https://pzp-hardware.readthedocs.io
# Licensed under the Apache License 2.0 - https://github.com/jdranczewski/pzp-hardware/blob/main/LICENSE

r"""
:module_title:`Nireos GEMINI-X hyperspectral camera`

Pieces for the `Nireos <https://www.nireos.com>`__ GEMINI-X interferometer and its camera
(e.g. a Basler a2A1280-125umSWIR), using the `puzzlepiece <https://puzzlepiece.readthedocs.io>`__
framework. The Hera SDK treats the interferometer and camera as a single "HERA device", which these
Pieces drive through the SDK's C API (``HeraAPI.dll``).

Example usage (see :ref:`getting-started` for more details on using Pieces in general)::

    # Import this before puzzlepiece/Qt, see the Installation notes below
    from pzp_hardware.nireos import hyperspectral_camera
    import puzzlepiece as pzp

    app = pzp.QApp()
    puzzle = pzp.Puzzle(debug=False)
    puzzle.add_piece("camera", hyperspectral_camera.Piece, row=0, column=0)
    puzzle.show()
    app.exec()

The Pieces have the main params and actions of :mod:`pzp_hardware.thorlabs.camera`
(serial, connected, armed, exposure, gain, black, roi, image, counts, background...), so a Puzzle
that uses a Thorlabs camera as e.g. its "spectrometer" can use this Piece instead. The Thorlabs
camera's unlimited, frame_buffer and Trigger have no equivalent on this camera and are left out.

The ``output`` param selects what the ``image`` param returns:

* ``"frame"``: a raw live camera frame, like a normal camera
* ``"spectral"``: a hyperspectral acquisition collapsed along one spatial axis, giving a
  (position x wavelength) image, like a slit spectrometer. ``image.shape[1]`` is the
  wavelength axis and the ``wls`` param holds the wavelength of each column in nm.

The full hypercube (wavelength, y, x) is available through the ``cube`` param, and "Scan Cube"
opens a window with the scan settings. "Save cube" writes the cube as a ``datasets`` .ds file
(only this action needs the ``datasets`` package).

Installation
------------
* Install the Nireos Hera SDK (version 1.3 or newer). The Pieces look for ``HeraAPI.dll`` in
  ``C:\Program Files\Nireos\Hera SDK\HeraAPI\bin``, and ask for the directory on first use if
  it's not there.
* Activate the SDK licence, either with ``C:\Program Files\Nireos\Hera SDK\License Manager\LicenseManager.exe``
  or with the "Activate licence" action in the Piece's settings (online activation needs internet access).
* ``HeraAPI.dll`` needs a recent MSVC C++ runtime (``msvcp140.dll`` 14.40 or newer), but PyQt5 bundles
  an older one. Whichever is loaded first is used by the whole process, so if Qt is loaded first the SDK
  crashes. Importing this module before puzzlepiece/Qt avoids this. Alternatively, preload the system
  runtime at Python startup by adding a file such as ``msvc_runtime_preload.pth`` to your environment's
  ``site-packages`` folder, containing the line::

    import os, ctypes; exec("try:\n ctypes.WinDLL(os.path.join(os.environ.get('SystemRoot', r'C:\\Windows'), 'System32', 'msvcp140.dll'))\nexcept OSError:\n pass")

Requirements
------------
.. pzp_requirements:: pzp_hardware.nireos.hyperspectral_camera

Available Pieces
----------------
"""
import os
import re
import sys
import threading
import time

# Imported before Qt, so the Hera SDK gets the MSVC runtime it needs (see _hera.py)
from pzp_hardware.nireos import _hera as hera

import numpy as np
import puzzlepiece as pzp
import pyqtgraph as pg
from puzzlepiece.extras import hardware_tools as pht
from pyqtgraph.Qt import QtCore, QtWidgets

pht.requirements(
    {
        "PIL": {
            "pip": "pillow",
            "url": "https://pillow.readthedocs.io/en/stable/installation/basic-installation.html",
        }
    }
)
from PIL import Image


# Labels shown in the GUI. The param names themselves are unchanged, so scripts keep using
# e.g. puzzle["spectrometer:scan_mode"] (and the names shared with pzp_hardware.thorlabs.camera still match).
LABELS = {
    "roi": "ROI",
    "exposure": "Exposure (ms)",
    "actual_gain": "Actual gain (dB)",
    "temperature": "Temperature (°C)",
    "stabilisation": "Stabilisation (ms)",
    "trigger_delay": "Trigger delay (µs)",
    "wls": "Wavelengths (nm)",
}


# The Hera SDK's default number of output bands (GetDefaultOutBands gives 120 for every scan
# mode of GVS2602006). It's shown before connecting, and replaced by the SDK's value on connect.
DEFAULT_BANDS = 120
# Used if the SDK has no default for the scan mode
FALLBACK_BANDS = 650


# Characters that can't go in a file name
_FILENAME_UNSAFE = re.compile(r'[<>:"/\\|?*{}\s]+')


def _label(name):
    """GUI label for a param name: spaces, and a capital on the first word only."""
    return LABELS.get(name, name.replace("_", " ").capitalize())


def _relabel(params):
    for name, param in params.items():
        param.label.setText(_label(name) + ":")


class _Settings(pzp.piece.Popup):
    def define_params(self):
        self.add_child_params((
            "black", "pixel_format", "flip_x", "flip_y", "counts", "max_counts", "sub_background",
            "actual_gain", "temperature",
        ))
        _relabel(self.params)
        return super().define_params()

    def define_actions(self):
        self.add_child_actions(("Take background", "ROI", "Reset ROI", "Rediscover", "Activate licence"))
        return super().define_actions()


class _SpectralSettings(pzp.piece.Popup):
    def define_params(self):
        self.add_child_params((
            "scan_mode", "averages", "stabilisation", "trigger_mode", "trigger_delay",
            "resource_usage", "spectral_axis",
        ))
        _relabel(self.params)
        return super().define_params()

    def define_actions(self):
        self.add_child_actions(("Acquire cube",))

        @pzp.action.define(self, "Take background")
        def take_background(self):
            return self.parent_piece.actions["Take background cube"]()

        self.add_child_actions(("Save cube", "Export ENVI"))
        return super().define_actions()

    def custom_layout(self):
        layout = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel("Warning")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        warning = QtWidgets.QLabel("Background cubes are only valid for the same scan mode, gain and exposure.")
        warning.setWordWrap(True)
        warning.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(warning)

        # Progress of the current acquisition. A plain progress bar is used as puzzlepiece
        # can't make a child param from a progress param once it holds a float value.
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        layout.addWidget(self.progress)

        def update_progress():
            value = self.parent_piece["cube_progress"].value
            self.progress.setValue(int((value or 0) * 1000))

        self._update_progress = pzp.threads.CallLater(update_progress)
        self.parent_piece["cube_progress"].changed.connect(self._update_progress)
        update_progress()

        # Plot of the mean spectrum of the latest hypercube
        self.pw = pg.PlotWidget()
        self.pw.setLabel("bottom", "wavelength (nm)")
        self.pw.setLabel("left", "mean counts")
        layout.addWidget(self.pw)
        self.line = self.pw.plot()

        def update_plot():
            cube, wls = self.parent_piece["cube"].value, self.parent_piece["wls"].value
            if cube is not None and wls is not None and len(wls) == cube.shape[0]:
                self.line.setData(wls, cube.mean(axis=(1, 2)))

        self._update_later = pzp.threads.CallLater(update_plot)
        self.parent_piece["cube"].changed.connect(self._update_later)
        update_plot()
        return layout

    def handle_close(self, event=None):
        self.parent_piece["cube"].changed.disconnect(self._update_later)
        self.parent_piece["cube_progress"].changed.disconnect(self._update_progress)


class Base(pzp.Piece):
    """
    Hyperspectral camera Piece without an image preview. Can be used to take frames and cubes
    and show settings without an explicit image view in the UI.

    Connecting in debug mode connects to a simulated device, which takes frames and quick
    fake hyperspectral scans.
    """

    # Move the custom_layout to the right of the generated inputs
    custom_horizontal = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _relabel(self.params)
        # self.image will store the image the camera takes
        self.image = None
        self.device = None
        # Sensor region (x, y, w, h) and settings the latest cube was acquired with
        self._cube_roi = None
        self._cube_settings = {}
        # Whether the latest cube is a background, and where the last background cube was saved
        self._cube_is_background = False
        self._background_file = None
        # Held while frames/cubes are acquired (possibly in the Live thread), so dispose() never
        # releases the device in the middle of an SDK call
        self._device_lock = threading.RLock()

    def define_params(self):
        # region connection
        @pzp.param.dropdown(self, "serial", "")
        def get_serials(self):
            if self.puzzle.debug:
                return None
            try:
                return self.puzzle.globals["hera_sdk"].discover_serials()
            except Exception as e:
                # Most likely the SDK licence is not activated
                print(f"hyperspectral_camera: could not list HERA devices: {e}")
                return None

        @pzp.param.checkbox(self, "connected", 0)
        def connect(self, value):
            current_value = self.params["connected"].value
            if value and not current_value:
                try:
                    if self.puzzle.debug:
                        self.device = hera.FakeHeraDevice()
                    else:
                        self.device = self.puzzle.globals["hera_sdk"].open_device(self["serial"].value or None)
                    # Take the SDK's default number of bands, unless it was already changed
                    if self["bands"].value == DEFAULT_BANDS:
                        self["bands"].set_value(self._sdk_default_bands())
                    return 1
                except Exception as e:
                    self.dispose()
                    raise e
            elif not value and current_value:
                self.dispose()
                return 0
            return current_value

        # endregion

        # region camera settings
        # Live stream on/off. Not shown in the GUI (it follows image requests and scans by itself),
        # kept for scripts written for pzp_hardware.thorlabs.camera.
        @pzp.param.checkbox(self, "armed", 0, visible=False)
        @self._ensure_connected
        def armed(self, value):
            current_value = self.params["armed"].value
            if value and not current_value:
                if self.device.is_acquiring_cube:
                    raise Exception("A hyperspectral acquisition is running")
                self.device.start_live(self["pixel_format"].value)
                return 1
            elif not value and current_value:
                self.device.stop_live()
                return 0
            return current_value

        # The exposure is set in ms, like pzp_hardware.thorlabs.camera (the SDK uses us)
        @pzp.param.spinbox(self, "exposure", 25.0, v_min=0.0)
        @self._ensure_connected
        def exposure(self, value):
            self.device.exposure_us = value * 1000

        @exposure.set_getter(self)
        @self._ensure_connected
        def get_exposure(self):
            return self.device.exposure_us / 1000

        # Hera exposes gain as a "gain level" from 0 to 1, see actual_gain for the resulting gain in dB
        @pzp.param.spinbox(self, "gain", 0.0, v_min=0.0, v_max=1.0, v_step=0.01)
        @self._ensure_connected
        def gain(self, value):
            self.device.gain_level = value

        @gain.set_getter(self)
        @self._ensure_connected
        def gain(self):
            return self.device.gain_level

        @pzp.param.readout(self, "actual_gain", False)
        @self._ensure_connected
        def actual_gain(self):
            return self.device.actual_gain

        @pzp.param.spinbox(self, "black", 0.0, visible=False)
        @self._ensure_connected
        def black(self, value):
            if not self.device.black_level_supported:
                raise Exception("Setting the black level is not supported by this camera")
            self.device.black_level = value

        @black.set_getter(self)
        @self._ensure_connected
        def black(self):
            return self.device.black_level

        @pzp.param.dropdown(self, "pixel_format", "Mono12", visible=False)
        def pixel_format(self):
            return hera.PIXEL_FORMATS

        @pixel_format.set_setter(self)
        @self._ensure_disarmed
        def pixel_format(self, value):
            if value not in hera.PIXEL_FORMATS:
                raise ValueError(f"pixel_format must be one of {hera.PIXEL_FORMATS}")
            if self.device is not None and not self.device.pixel_format_supported(value, False):
                raise Exception(f"{value} is not supported by this camera")

        @pzp.param.readout(self, "temperature", False)
        @self._ensure_connected
        def temperature(self):
            return self.device.temperature[1]

        # The roi is [x1, y1, x2, y2] (inclusive) in native sensor coordinates, like pzp_hardware.thorlabs.camera.
        # The camera rounds x1 and the width up to multiples of 4, the value read back is the actual roi.
        @pzp.param.array(self, "roi", False)
        @self._ensure_connected
        def roi(self):
            x, y, w, h = self.device.roi
            return [x, y, x + w - 1, y + h - 1]

        @roi.set_setter(self)
        @self._ensure_connected
        @self._ensure_disarmed
        def roi(self, value):
            x1, y1, x2, y2 = (int(v) for v in value)
            self.device.roi = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
            x, y, w, h = self.device.roi
            return [x, y, x + w - 1, y + h - 1]

        # Orientation of the returned images, to match how the camera is mounted
        pzp.param.checkbox(self, "flip_x", 0, visible=False)(None)
        pzp.param.checkbox(self, "flip_y", 0, visible=False)(None)
        # endregion

        # region spectral settings
        # The spectral resolution is set by the length of the interferometer scan
        @pzp.param.dropdown(self, "scan_mode", "Medium", visible=False)
        def scan_mode(self):
            return hera.SCAN_MODES

        @scan_mode.set_setter(self)
        def scan_mode(self, value):
            # ExtraLong is not supported by GVS2602006
            if self.device is not None and value not in self.device.supported_scan_modes:
                raise Exception(f"Scan mode {value} is not supported, use one of {self.device.supported_scan_modes}")
        # Number of output wavelength bands, 0 uses the SDK default for the scan mode. Only used
        # when the cube is computed from the scan data, so it can be changed with "Recompute cube".
        pzp.param.spinbox(self, "bands", DEFAULT_BANDS, v_min=0)(None)
        # Spatial binning of the cube (Bin2x halves its width and height), speeds up the FT
        pzp.param.dropdown(self, "binning", "Bin2x")(list(hera.BINNING))
        # Frames averaged at each interferometer step (scan time scales linearly with it)
        pzp.param.spinbox(self, "averages", 1, v_min=1, visible=False)(None)
        # Wait between interferometer steps, in ms, for setups disturbed by the mechanical movement
        pzp.param.spinbox(self, "stabilisation", 0, v_min=0, visible=False)(None)
        pzp.param.dropdown(self, "trigger_mode", "Internal", visible=False)(hera.TRIGGER_MODES)
        # The spatial axis summed over to make the (position x wavelength) spectral image
        pzp.param.dropdown(self, "spectral_axis", "x", visible=False)(["x", "y"])
        pzp.param.dropdown(self, "output", "frame")(["frame", "spectral"])

        # External trigger delay in us
        @pzp.param.spinbox(self, "trigger_delay", 0.0, v_min=0.0, visible=False)
        @self._ensure_connected
        def trigger_delay(self, value):
            if not self.device.external_trigger_delay_supported:
                raise Exception("External trigger delay is not supported by this device")
            self.device.external_trigger_delay_us = value

        @trigger_delay.set_getter(self)
        @self._ensure_connected
        def trigger_delay(self):
            if not self.device.external_trigger_delay_supported:
                return self["trigger_delay"].value
            return self.device.external_trigger_delay_us

        # CPU/memory the SDK may use (mostly for the FT). The SDK starts at MaxUsage.
        @pzp.param.dropdown(self, "resource_usage", "MaxUsage", visible=False)
        def resource_usage(self):
            return hera.RESOURCE_USAGE

        @resource_usage.set_setter(self)
        def resource_usage(self, value):
            if value not in hera.RESOURCE_USAGE:
                raise ValueError(f"resource_usage must be one of {hera.RESOURCE_USAGE}")
            if not self.puzzle.debug:
                self.puzzle.globals["hera_sdk"].set_resource_usage(value)

        pzp.param.progress(self, "cube_progress", visible=False)(None)

        @pzp.param.array(self, "cube", False)
        @self._ensure_connected
        def cube(self):
            return self._acquire_cube()

        # Wavelengths (nm) of the latest cube bands, i.e. of the spectral image columns
        pzp.param.array(self, "wls", False)(None)

        # Mean spectrum over the ROI, from a new hyperspectral acquisition
        @pzp.param.array(self, "spectrum", False)
        def spectrum(self):
            return self["cube"].get_value().mean(axis=(1, 2))

        # endregion

        # region image
        @pzp.param.array(self, "image")
        @self._ensure_connected
        def get_image(self):
            if self["output"].value == "spectral":
                image = self._spectral_image(self["cube"].get_value())
            else:
                image = self._orient(self._grab_frame())
            return self._subtract_background(image)

        @pzp.param.readout(self, "counts", False)
        def get_counts(self):
            image = self.params["image"].get_value()
            return np.sum(image)

        @pzp.param.readout(self, "max_counts", False)
        def get_max_counts(self):
            image = self.params["image"].get_value()
            return np.amax(image)

        pzp.param.checkbox(self, "sub_background", 0, visible=False)(None)
        pzp.param.array(self, "background", False)(None)
        # endregion

    def define_actions(self):
        @pzp.action.define(self, "Take background", visible=False)
        def take_background(self):
            sub = self["sub_background"].value
            self["sub_background"].set_value(0)
            try:
                self["background"].set_value(self["image"].get_value().astype(float))
            finally:
                self["sub_background"].set_value(sub)

        @pzp.action.define(self, "ROI", visible=False)
        @self._ensure_connected
        def roi(self):
            self.open_popup(_ROI_Popup, "Region of interest")

        @pzp.action.define(self, "Reset ROI", visible=False)
        @self._ensure_connected
        @self._ensure_disarmed
        def reset_roi(self):
            self.device.clear_roi()
            self["roi"].get_value()

        @pzp.action.define(self, "Save image")
        def save_image(self, filename=None):
            image = self.params["image"].value
            if image is None:
                image = self.params["image"].get_value()

            if filename is None:
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self.puzzle, "Save file as...", ".", "Image files (*.png)")
                if not filename:
                    return

            if image.dtype == np.uint8:
                Image.fromarray(image).save(filename)
            elif image.dtype == np.uint16:
                Image.fromarray(image, mode="I;16").save(filename)
            else:
                # Float images (background-subtracted or spectral) are scaled to 16 bits
                span = np.ptp(image) or 1
                Image.fromarray(((image - image.min()) / span * 65535).astype(np.uint16), mode="I;16").save(filename)

        @pzp.action.define(self, "Rediscover", visible=False)
        def rediscover(self):
            if not self.puzzle.debug:
                serials = self.puzzle.globals["hera_sdk"].discover_serials()
                self.params["serial"].input.clear()
                self.params["serial"].input.addItems(serials)

        @pzp.action.define(self, "Acquire cube", visible=False)
        def acquire_cube(self):
            return self["cube"].get_value()

        # Compute the cube again from the raw data of the latest acquisition, using the current
        # bands / binning (no new scan)
        @pzp.action.define(self, "Recompute cube", visible=False)
        @self._ensure_connected
        def recompute_cube(self):
            wls, cube = self.device.recompute_cube(self["bands"].value, self["binning"].value)
            self._cube_settings.update(self._record(("bands", "binning")))
            self["wls"].set_value(wls)
            self["cube"].set_value(cube)
            if self["output"].value == "spectral":
                self["image"].set_value(self._subtract_background(self._spectral_image(cube)))
            return cube

        # Saves the latest cube (acquiring one if there is none) as a `datasets` .ds file:
        # data.raw[i] is the image at wavelength data.wl[i] (nm), and data.y / data.x
        # are sensor pixel coordinates (centres of binned pixels, native orientation - no flips).
        # data.metadata holds the settings the cube was taken with, keyed like puzzle.record_values.
        @pzp.action.define(self, "Save cube", visible=False)
        def save_cube(self, filename=None):
            pht.requirements(["datasets"])
            import datasets as ds

            cube, wls = self["cube"].value, self["wls"].value
            if cube is None:
                cube, wls = self["cube"].get_value(), self["wls"].value
            if filename is None:
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self.puzzle, "Save cube as...", ".", "Dataset (*.ds)")
                if not filename:
                    return
            n_rows, n_cols = cube.shape[1:]
            x0, y0, w, h = self._cube_roi or (0, 0, n_cols, n_rows)
            data = ds.dataset(
                cube, wl=np.asarray(wls),
                y=y0 + (np.arange(n_rows) + 0.5) * h / n_rows - 0.5,
                x=x0 + (np.arange(n_cols) + 0.5) * w / n_cols - 0.5,
            )
            data.metadata.update(self._cube_settings)
            data.metadata["cube_roi"] = self._cube_roi
            # For post-processing: backgrounds are not subtracted here
            data.metadata["cube_type"] = "background" if self._cube_is_background else "signal"
            data.metadata["background_file"] = None if self._cube_is_background else self._background_file
            data.metadata["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            data.save(filename)
            print(f"Saved {filename}  {data!r}")
            return filename

        # Scan a background cube (source blocked, room lights as for the measurement) and save it
        # as a .ds file. It becomes the current cube, but it is not subtracted from later cubes:
        # that is left to post-processing, using the background_file in their metadata.
        @pzp.action.define(self, "Take background cube", visible=False)
        def take_background_cube(self, filename=None):
            self["cube"].get_value()
            self._cube_is_background = True
            if filename is None:
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self.puzzle, "Save background cube as...", self._background_filename(), "Dataset (*.ds)")
                if not filename:
                    return None
            self._background_file = os.path.abspath(self.actions["Save cube"](filename))
            return self._background_file

        @pzp.action.define(self, "Export ENVI", visible=False)
        @self._ensure_connected
        def export_envi(self, filename=None):
            # Exports the SDK hypercube in ENVI format
            if filename is None:
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self.puzzle, "Export ENVI cube as...", ".", "ENVI (*.hdr *.raw);;All files (*)")
                if not filename:
                    return
            self.device.export_envi(filename, "Exported by pzp_hardware.nireos.hyperspectral_camera")

        @pzp.action.define(self, "Activate licence", visible=False)
        def activate_licence(self, key=None):
            if self.puzzle.debug:
                return
            if key is None:
                key, ok = QtWidgets.QInputDialog.getText(self.puzzle, "Hera SDK licence", "Licence key:")
                if not ok or not key:
                    return
            sdk = self.puzzle.globals["hera_sdk"]
            sdk.activate_license_online(key.strip())
            print(f"hyperspectral_camera: Hera SDK licensed: {sdk.license_status()[0]}")
            self.actions["Rediscover"]()

        @pzp.action.define(self, "Settings")
        def settings(self):
            self.open_popup(_Settings, "Camera settings")

        # Opens the scan window: scan settings, progress, and the button that acquires a cube
        @pzp.action.define(self, "Scan Cube")
        def scan_cube(self):
            return self.open_popup(_SpectralSettings, "Scan cube")

    # Main panel: (section title, params) in display order. Visible params not listed here
    # (e.g. added by a subclass) are shown at the end.
    PANEL_LAYOUT = (
        (None, ("serial", "connected", "exposure", "gain", "view")),
        ("Image display", ("image", "autolevel")),
        ("Cube display", ("bands", "binning", "output")),
    )

    def param_layout(self, wrap=None):
        layout = QtWidgets.QVBoxLayout()
        listed = set()
        for title, names in self.PANEL_LAYOUT:
            names = [name for name in names if name in self.params and self.params[name].visible]
            if title and names:
                label = QtWidgets.QLabel(title)
                label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                font = label.font()
                font.setBold(True)
                label.setFont(font)
                label.setContentsMargins(0, 6, 0, 0)
                layout.addWidget(label)
            for name in names:
                layout.addWidget(self.params[name])
            listed.update(names)
        for name, param in self.params.items():
            if param.visible and name not in listed:
                layout.addWidget(param)
        return layout

    def action_layout(self, wrap=None):
        # As puzzlepiece's, but with a larger "Scan Cube" button on its own row at the top
        layout = QtWidgets.QGridLayout()
        scan = QtWidgets.QPushButton("Scan Cube")
        scan.setMinimumHeight(40)
        font = scan.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.2)
        scan.setFont(font)
        scan.clicked.connect(lambda: self.actions["Scan Cube"]())
        layout.addWidget(scan, 0, 0, 1, self.action_wrap)
        others = [key for key in self.actions if self.actions[key].visible and key != "Scan Cube"]
        for i, key in enumerate(others):
            button = QtWidgets.QPushButton(key)
            button.clicked.connect(lambda x=False, _key=key: self.actions[_key]())
            layout.addWidget(button, 1 + i // self.action_wrap, i % self.action_wrap)
        return layout

    # region helpers
    def _sdk_default_bands(self):
        """The SDK's default number of bands for the current scan mode (or FALLBACK_BANDS)."""
        try:
            bands = self.device.default_out_bands(self["scan_mode"].value)
        except Exception:
            bands = 0
        return bands if bands > 0 else FALLBACK_BANDS

    def _background_filename(self):
        """BACKGROUND_{sample:date}_{sample:sample}_{sample:network}_{scan mode}.ds, with blank
        fields if the Puzzle has no sample piece."""
        fields = []
        for name in ("date", "sample", "network"):
            try:
                value = str(self.puzzle["sample"][name].value or "")
            except Exception:
                value = ""
            fields.append(_FILENAME_UNSAFE.sub("-", value).strip("-_"))
        return "_".join(["BACKGROUND", *fields, self["scan_mode"].value]) + ".ds"

    def _record(self, names):
        """{"<piece>:<param>": value} for these params, like puzzle.record_values."""
        piece = self._name or "hyperspectral_camera"
        return {f"{piece}:{name}": self[name].value for name in names}

    def _grab_frame(self):
        """Return a live frame in native sensor orientation (arming the camera if needed)."""
        with self._device_lock:
            if self.device is None:
                raise Exception("Camera not connected")
            self._ensure_armed()
            # Wait for a frame whose exposure started after this call, not an older one in the stream
            return self.device.get_frame(fresh=True, timeout=max(5.0, 3 * self.device.exposure_us * 1e-6 + 2))

    def _orient(self, image):
        """Apply flip_x/flip_y to an array whose last two axes are (y, x)."""
        if self["flip_x"].value:
            image = image[..., ::-1]
        if self["flip_y"].value:
            image = image[..., ::-1, :]
        return image

    def _spectral_image(self, cube):
        """Collapse a (wavelength, y, x) cube to a (position, wavelength) image."""
        cube = self._orient(cube)
        axis = 2 if self["spectral_axis"].value == "x" else 1
        return cube.sum(axis=axis).T

    def _subtract_background(self, image):
        if self["sub_background"].value:
            background = self["background"].value
            if background is None or background.shape != image.shape:
                raise Exception("Background missing or of the wrong shape, take a new one")
            image = image.astype(float) - background
        return image

    def _in_main_thread(self):
        return QtCore.QThread.currentThread() == QtWidgets.QApplication.instance().thread()

    def _acquire_cube(self):
        """Run a hyperspectral acquisition, returning the (wavelength, y, x) cube."""
        with self._device_lock:
            return self._acquire_cube_locked()

    def _acquire_cube_locked(self):
        self._ensure_disarmed()
        self.stop = False
        self._cube_is_background = False
        progress = self["cube_progress"]
        # Read back from the camera, so the metadata has the values actually used
        for name in ("exposure", "gain", "actual_gain", "temperature"):
            self[name].get_value()
        settings = self._record((
            "serial", "exposure", "gain", "actual_gain", "pixel_format", "temperature",
            "scan_mode", "averages", "stabilisation", "trigger_mode", "trigger_delay", "flip_x", "flip_y",
        ))
        self.device.start_cube(
            self["scan_mode"].value, self["trigger_mode"].value,
            self["averages"].value, self["stabilisation"].value,
        )
        progress.set_value(0)
        while not self.device.cube_done:
            if self.stop:
                self.device.abort_cube()
                self.stop = False
                raise Exception("Hyperspectral acquisition aborted")
            progress.set_value(self.device.cube_progress)
            if self._in_main_thread():
                self.puzzle.process_events()
                if self.device is None:
                    # The window was closed (or the camera disconnected) while processing events
                    raise Exception("Camera disconnected during the hyperspectral acquisition")
            time.sleep(0.05)
        progress.set_value(1)

        self._cube_roi = tuple(self.device.roi)
        wls, cube = self.device.finish_cube(self["bands"].value, self["binning"].value)
        settings.update(self._record(("bands", "binning")))
        self._cube_settings = settings
        self["wls"].set_value(wls)
        return cube

    # endregion

    @pzp.piece.ensurer
    def _ensure_connected(self):
        if not self.params["connected"].value or self.device is None:
            raise Exception("Camera not connected")

    @pzp.piece.ensurer
    def _ensure_armed(self):
        if not self.params["armed"].value:
            self.params["armed"].set_value(1)

    @pzp.piece.ensurer
    def _ensure_disarmed(self):
        if self.params["armed"].value:
            self.params["armed"].set_value(0)

    def setup(self):
        # Called only if not in debug mode, sets up the Hera SDK (shared between Pieces)
        if not self.puzzle.globals.require("hera_sdk"):
            dll_directory = hera.DEFAULT_DLL_DIR
            if not os.path.exists(os.path.join(dll_directory, "HeraAPI.dll")):
                # The SDK is installed somewhere else, ask the user where (remembered in a .config file)
                dll_directory = pht.config(
                    "hera_dll_directory",
                    default=dll_directory,
                    description="the folder containing HeraAPI.dll (Hera SDK\\HeraAPI\\bin)",
                    validator=pht.validator_path_exists,
                )
            sdk = hera.HeraSDK(dll_directory)
            problem = hera.runtime_problem()
            if problem:
                print(f"hyperspectral_camera: {problem}")
            licensed, expiry, _ = sdk.license_status()
            if not licensed:
                print("hyperspectral_camera: the Hera SDK licence is not activated. Use 'Activate licence' in the camera "
                      "settings, or 'C:\\Program Files\\Nireos\\Hera SDK\\License Manager\\LicenseManager.exe'.")
            self.puzzle.globals["hera_sdk"] = sdk

    def dispose(self):
        # This function 'disposes' of the device, effectively disconnecting us
        if self.device is not None:
            # Wait for a frame or cube being acquired in another thread to finish (call_stop has
            # already asked it to stop), but don't hang forever
            locked = self._device_lock.acquire(timeout=10)
            try:
                if self.device is not None:
                    self.device.close()
            finally:
                self.device = None
                if locked:
                    self._device_lock.release()
                self.params["armed"].set_value(0, skip_setter=True)
                self.params["connected"].set_value(0, skip_setter=True)

    def call_stop(self):
        super().call_stop()

    def handle_close(self, event):
        if self.puzzle.debug:
            return
        try:
            self.dispose()
        finally:
            if self.puzzle.globals.release("hera_sdk"):
                # If we are the last Piece using the SDK, we shut the SDK down
                self.puzzle.globals["hera_sdk"].cleanup()
                del self.puzzle.globals["hera_sdk"]


class _ROI_Popup(pzp.piece.Popup):
    """Set the ROI interactively on a full-sensor reference frame (native sensor orientation)."""

    def define_actions(self):
        @pzp.action.define(self, "set_roi_from_camera", visible=False)
        def set_roi_from_camera(self):
            camera_roi = self.parent_piece.params["roi"].get_value()
            self.roi_item.setPos(camera_roi[:2])
            self.roi_item.setSize([camera_roi[2] - camera_roi[0] + 1, camera_roi[3] - camera_roi[1] + 1])

        @pzp.action.define(self, "Capture ref")
        def capture_reference(self):
            parent = self.parent_piece
            original_roi = parent.params["roi"].get_value()
            parent.actions["Reset ROI"]()
            try:
                image = parent._grab_frame()
            finally:
                parent.params["armed"].set_value(0)
                parent.params["roi"].set_value(original_roi)
            self._rows, self._cols = image.shape
            self.imgw.setImage(image)

        @pzp.action.define(self, "Set ROI")
        def set_roi(self):
            x1, y1 = self.roi_item.pos()
            x2, y2 = self.roi_item.size()
            x2 += x1 - 1
            y2 += y1 - 1
            x1, x2, y1, y2 = (int(np.round(x)) for x in (x1, x2, y1, y2))
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, self._cols - 1)
            y2 = min(y2, self._rows - 1)
            self.parent_piece.params["armed"].set_value(0)
            self.parent_piece.params["roi"].set_value((x1, y1, x2, y2))
            self.actions["set_roi_from_camera"]()

        @pzp.action.define(self, "Centre")
        def centre_roi(self):
            size = self.roi_item.size()
            self.roi_item.setPos((self._cols / 2 - size[0] / 2, self._rows / 2 - size[1] / 2))

        @pzp.action.define(self, "Reset")
        def reset_roi(self):
            self.roi_item.setPos((0, 0))
            self.roi_item.setSize((self._cols, self._rows))

    def custom_layout(self):
        layout = QtWidgets.QVBoxLayout()

        self.pw = pg.PlotWidget()
        layout.addWidget(self.pw)

        plot_item = self.pw.getPlotItem()
        plot_item.setAspectLocked(True)
        plot_item.invertY(True)
        plot_item.showGrid(True, True)

        self.imgw = pg.ImageItem(border="w", axisOrder="row-major")
        plot_item.addItem(self.imgw)

        self.roi_item = pg.ROI([0, 0], [10, 10], pen=(255, 255, 0, 200))
        self.roi_item.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi_item.addScaleHandle([1, 0.5], [0.5, 0.5])
        self.actions["set_roi_from_camera"]()
        self.actions["Capture ref"]()
        self.imgw.update()
        plot_item.addItem(self.roi_item)

        return layout


class Piece(Base):
    """
    Like :class:`~pzp_hardware.nireos.hyperspectral_camera.Base`, but with a live view. ``view``
    selects what is displayed: the ``image`` param, or the band of the latest cube picked with the
    wavelength slider above the image. It follows whatever changed last (a new image shows the
    image, a new cube or moving the slider shows the cube). The plot below the image shows the
    cube spectrum of the pixel under the mouse.

    .. image:: ../images/pzp_hardware.nireos.hyperspectral_camera.Piece.png
    """

    def define_params(self):
        super().define_params()

        @pzp.param.checkbox(self, "autolevel", 0)
        def autolevel(self, value):
            if self["view"].value == "cube" and hasattr(self, "_redraw"):
                self._redraw()
                return
            image = self["image"].value
            if value and image is not None:
                self.imgw.setLevels([np.amin(image), np.amax(image)])
            else:
                self.imgw.setLevels(self._default_levels())

        pzp.param.dropdown(self, "view", "image")(["image", "cube"])

    def _default_levels(self):
        pixel_format = self["pixel_format"].value
        bits = int(pixel_format[4:]) if pixel_format in hera.PIXEL_FORMATS else 12
        return [0, 2 ** bits - 1]

    @staticmethod
    def _show(imgw, image, levels=None):
        """
        Display an image with the given levels (the image's own range if None). The levels are set
        after setImage, as pyqtgraph applies levels set before an ImageItem had any image after its
        first setImage, which would override levels passed to that call.
        """
        imgw.setImage(image, autoLevels=False)
        if levels is None:
            low, high = float(np.amin(image)), float(np.amax(image))
            levels = [low, high if high > low else low + 1]
        imgw.setLevels(levels)

    def _set_image(self, imgw, plot_item):
        """Display the current image, with a wavelength x axis in spectral mode."""
        image = self["image"].value
        if image is None:
            return
        wls = self["wls"].value
        autolevel = self["autolevel"].value
        if self["output"].value == "spectral" and wls is not None and len(wls) == image.shape[1] and len(wls) > 1:
            self._show(imgw, image, None if autolevel or image.dtype.kind == "f" else self._default_levels())
            step = (wls[-1] - wls[0]) / (len(wls) - 1)
            imgw.setRect(QtCore.QRectF(wls[0] - step / 2, 0, wls[-1] - wls[0] + step, image.shape[0]))
            plot_item.setAspectLocked(False)
            plot_item.setLabel("bottom", "wavelength (nm)")
        else:
            self._show(imgw, image, None if autolevel else self._default_levels())
            imgw.setRect(QtCore.QRectF(0, 0, image.shape[1], image.shape[0]))
            plot_item.setAspectLocked(True)
            plot_item.setLabel("bottom", "")

    # region cube view
    def _cube_ok(self):
        cube, wls = self["cube"].value, self["wls"].value
        return cube is not None and wls is not None and np.ndim(cube) == 3 and len(wls) == cube.shape[0]

    def _band_index(self):
        return min(self.wl_slider.value(), len(self["wls"].value) - 1)

    def _set_band(self, plot_item):
        """Display the cube band selected by the slider, drawn over the sensor region it was taken of."""
        if not self._cube_ok():
            return
        cube = self["cube"].value
        band = self._orient(cube[self._band_index()])
        self._show(self.imgw, band, None if self["autolevel"].value else self._cube_levels)
        # Binned cubes are stretched to the size of the region, so pixels line up with live frames
        w, h = self._cube_roi[2:] if self._cube_roi else (band.shape[1], band.shape[0])
        self.imgw.setRect(QtCore.QRectF(0, 0, w, h))
        plot_item.setAspectLocked(True)
        plot_item.setLabel("bottom", "")

    def _update_wl_label(self):
        if self._cube_ok():
            wls = self["wls"].value
            i = self._band_index()
            self.wl_label.setText(f"{wls[i]:.1f} nm")
            self.spectrum_marker.setValue(wls[i])
        else:
            self.wl_label.setText("no cube")

    def _on_new_cube(self):
        """New cube: update the slider range (keeping the wavelength) and the colour levels."""
        if not self._cube_ok():
            return
        cube, wls = self["cube"].value, self["wls"].value
        self.wl_slider.blockSignals(True)
        self.wl_slider.setRange(0, len(wls) - 1)
        if self._slider_wl is not None:
            self.wl_slider.setValue(int(np.argmin(np.abs(wls - self._slider_wl))))
        else:
            # Start on the brightest band
            self.wl_slider.setValue(int(np.argmax(cube.mean(axis=(1, 2)))))
        self.wl_slider.blockSignals(False)
        self._slider_wl = wls[self._band_index()]
        low, high = np.percentile(cube[:, ::4, ::4], (0.1, 99.9))
        self._cube_levels = [float(low), float(high if high > low else low + 1)]
        self._update_wl_label()
        self._hover_spectrum(self._last_hover)

    def _cube_pixel(self, vx, vy):
        """
        Map image view coordinates to (row, col) of the cube and sensor (x, y) or None if the
        point is outside the cube. The view shows either a live frame of the current roi, or a
        cube band stretched over the roi the cube was taken of; both may be flipped.
        """
        if not self._cube_ok() or self._cube_roi is None:
            return None
        if self["view"].value == "cube":
            x0, y0, w, h = self._cube_roi
        else:
            roi = self["roi"].value
            if roi is None:
                return None
            x0, y0, w, h = roi[0], roi[1], roi[2] - roi[0] + 1, roi[3] - roi[1] + 1
        if not (0 <= vx < w and 0 <= vy < h):
            return None
        sx = x0 + (w - vx if self["flip_x"].value else vx)
        sy = y0 + (h - vy if self["flip_y"].value else vy)
        cx0, cy0, cw, ch = self._cube_roi
        n_rows, n_cols = self["cube"].value.shape[1:]
        col = int((sx - cx0) * n_cols / cw)
        row = int((sy - cy0) * n_rows / ch)
        if 0 <= row < n_rows and 0 <= col < n_cols:
            return row, col, int(sx), int(sy)
        return None

    def _hover_spectrum(self, view_pos):
        """Plot the spectrum at an image view position (a QPointF in view coordinates)."""
        self._last_hover = view_pos
        if view_pos is None:
            return
        vx, vy = view_pos.x(), view_pos.y()
        wls = self["wls"].value
        image = self["image"].value
        if self["view"].value == "image" and self["output"].value == "spectral":
            # The spectral image is (position, wavelength): show the row under the mouse
            if image is None or wls is None or np.ndim(image) != 2 or image.shape[1] != len(wls):
                return
            row = int(vy)
            if 0 <= row < image.shape[0]:
                self.spectrum_line.setData(wls, image[row])
                self.spectrum_plot.setTitle(f"position {row}", size="9pt")
            return
        pixel = self._cube_pixel(vx, vy)
        if pixel is None:
            return
        row, col, sx, sy = pixel
        self.spectrum_line.setData(wls, self["cube"].value[:, row, col])
        self.spectrum_plot.setTitle(f"sensor x={sx}, y={sy}", size="9pt")

    # endregion

    # Within this function we can define any other GUI objects we want to display beyond the
    # params and actions (which are generated by default)
    def custom_layout(self):
        layout = QtWidgets.QVBoxLayout()

        # Add a PuzzleTimer for live view
        delay = 0.05 if not self.puzzle.debug else 0.1
        self.timer = pzp.threads.PuzzleTimer("Live", self.puzzle, self.params["image"].get_value, delay)
        layout.addWidget(self.timer)

        # Wavelength slider, selects the cube band shown when view is "cube"
        self._slider_wl = None
        self._cube_levels = [0, 1]
        self._last_hover = None
        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(QtWidgets.QLabel("λ"))
        self.wl_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.wl_slider.setRange(0, 0)
        slider_row.addWidget(self.wl_slider)
        self.wl_label = QtWidgets.QLabel("no cube")
        self.wl_label.setMinimumWidth(70)
        slider_row.addWidget(self.wl_label)
        # New bands / binning from the latest scan's data, without scanning again
        recompute = QtWidgets.QPushButton("Recompute cube")
        recompute.clicked.connect(lambda: self.actions["Recompute cube"]())
        slider_row.addWidget(recompute)
        layout.addLayout(slider_row)

        self.pw = pg.PlotWidget()
        layout.addWidget(self.pw, stretch=3)

        plot_item = self.pw.getPlotItem()
        plot_item.setAspectLocked(True)
        plot_item.invertY(True)

        self.imgw = pg.ImageItem(border="w", axisOrder="row-major", levels=self._default_levels())
        plot_item.addItem(self.imgw)

        # Spectrum of the pixel under the mouse. The marker shows (and can drag) the slider wavelength.
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setMinimumHeight(150)
        self.spectrum_plot.setLabel("bottom", "wavelength (nm)")
        self.spectrum_plot.setTitle("hover over the image", size="9pt")
        self.spectrum_line = self.spectrum_plot.plot(pen="y")
        self.spectrum_marker = pg.InfiniteLine(angle=90, movable=True, pen=(100, 200, 255, 180))
        self.spectrum_plot.addItem(self.spectrum_marker)
        layout.addWidget(self.spectrum_plot, stretch=1)

        def redraw():
            if self["view"].value == "cube":
                self._set_band(plot_item)
            else:
                self._set_image(self.imgw, plot_item)

        self._redraw = redraw
        self["autolevel"].set_value(0)

        def show(view):
            if self["view"].value != view:
                self["view"].set_value(view)  # redraws through view.changed
            else:
                redraw()

        def on_image():
            show("image")
            self._hover_spectrum(self._last_hover)

        def on_cube():
            self._on_new_cube()
            show("cube")

        def on_slider(value):
            if self._cube_ok():
                self._slider_wl = self["wls"].value[self._band_index()]
            self._update_wl_label()
            show("cube")

        def on_marker():
            if self._cube_ok():
                wls = self["wls"].value
                self.wl_slider.setValue(int(np.argmin(np.abs(wls - self.spectrum_marker.value()))))

        def on_mouse(scene_pos):
            if plot_item.sceneBoundingRect().contains(scene_pos):
                self._hover_spectrum(plot_item.vb.mapSceneToView(scene_pos))

        self.params["image"].changed.connect(pzp.threads.CallLater(on_image))
        self.params["cube"].changed.connect(pzp.threads.CallLater(on_cube))
        self.params["view"].changed.connect(pzp.threads.CallLater(redraw))
        self.wl_slider.valueChanged.connect(on_slider)
        self.spectrum_marker.sigPositionChangeFinished.connect(on_marker)
        self.pw.scene().sigMouseMoved.connect(on_mouse)

        return layout

    def call_stop(self):
        self.timer.stop()
        super().call_stop()


class LineoutPiece(Piece):
    """
    Like :class:`~pzp_hardware.nireos.hyperspectral_camera.Base`, but the live view of the
    ``image`` param includes two movable lines (horizontal and vertical), and plots that show
    the image profile along them. A circle is shown where they cross. There is no cube view.
    """

    def define_actions(self):
        super().define_actions()

        @pzp.action.define(self, "Centre lines", QtCore.Qt.Key.Key_C)
        def centre(self):
            shape = self.params["image"].value.shape
            self._inf_line_x.setValue(shape[0] // 2)
            self._inf_line_y.setValue(shape[1] // 2)

    def define_params(self):
        super().define_params()
        pzp.param.spinbox(self, "circle_r", 200, visible=False)(None)

    def custom_layout(self):
        layout = QtWidgets.QVBoxLayout()

        self.timer = pzp.threads.PuzzleTimer("Live", self.puzzle, self.params["image"].get_value, 0.1)
        layout.addWidget(self.timer)

        self.gl = pg.GraphicsLayoutWidget()
        layout.addWidget(self.gl)
        self.gl.ci.layout.setRowStretchFactor(0, 2)
        self.gl.ci.layout.setColumnStretchFactor(0, 2)

        plot_main = self.gl.addPlot(0, 0)
        plot_x = self.gl.addPlot(1, 0)
        plot_x.setXLink(plot_main)
        plot_y = self.gl.addPlot(0, 1)
        plot_y.setYLink(plot_main)

        plot_main.setAspectLocked(True)
        plot_main.invertY(True)
        plot_y.invertY(True)

        self.imgw = pg.ImageItem(border="w", axisOrder="row-major", levels=self._default_levels())
        self["autolevel"].set_value(0)
        plot_main.addItem(self.imgw)

        self._inf_line_x = pg.InfiniteLine(0, 0, movable=True)
        self._inf_line_y = pg.InfiniteLine(0, 90, movable=True)
        plot_main.addItem(self._inf_line_x)
        plot_main.addItem(self._inf_line_y)

        r = self.params["circle_r"].value
        self._circle = QtWidgets.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
        self._circle.setPen(pg.mkPen((255, 255, 0, 150)))
        plot_main.addItem(self._circle)

        plot_line_x = plot_x.plot([0], [0])
        plot_line_y = plot_y.plot([0], [0])

        def update_image():
            image_data = self.params["image"].value
            if image_data is None:
                return
            self._set_image(self.imgw, plot_main)
            r = self.params["circle_r"].value
            self._circle.setRect(self._inf_line_y.value() - r, self._inf_line_x.value() - r, r * 2, r * 2)
            row = int(np.clip(self._inf_line_x.value(), 0, image_data.shape[0] - 1))
            col = int(np.clip(self._inf_line_y.value(), 0, image_data.shape[1] - 1))
            plot_line_x.setData(image_data[row])
            plot_line_y.setData(image_data[:, col], range(image_data.shape[0]))

        update_later = pzp.threads.CallLater(update_image)
        self.params["image"].changed.connect(update_later)
        self._inf_line_x.sigPositionChanged.connect(update_image)
        self._inf_line_y.sigPositionChanged.connect(update_image)

        return layout


def _debug_from_argv_or_env():
    """
    Choose debug mode without the interactive prompt if asked to:
    `python hyperspectral_camera.py --debug` / `--no-debug`, or the PZP_DEBUG=1/0 environment variable.
    Otherwise show the usual puzzlepiece debug prompt.
    """
    if "--debug" in sys.argv:
        return True
    if "--no-debug" in sys.argv:
        return False
    env = os.environ.get("PZP_DEBUG")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "n", "")
    return pht.debug_prompt()


if __name__ == "__main__":
    # If running this file directly, make a Puzzle, add our Piece, and display it
    from puzzlepiece.pieces import plotter

    app = pzp.QApp()
    puzzle = pzp.Puzzle(name="Hyperspectral camera", debug=_debug_from_argv_or_env())
    puzzle.add_piece("camera", Piece, 0, 0)
    puzzle.add_piece("plotter", plotter.Piece, 0, 1)
    puzzle["plotter"]["param"].set_value("camera:counts")
    puzzle.show()
    app.exec()
