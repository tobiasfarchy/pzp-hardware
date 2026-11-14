# This file is a part of pzp-hardware, a library of laboratory hardware support Pieces
# for the puzzlepiece GUI & automation framework. Check out https://pzp-hardware.readthedocs.io
# Licensed under the Apache License 2.0 - https://github.com/jdranczewski/pzp-hardware/blob/main/LICENSE

r"""
:module_title:`ThorLabs KDC101 rotation mount`

Pieces for interacting with a ThorLabs
`KDC101 K-Cube brushed DC servo controller <https://www.thorlabs.com/thorproduct.cfm?partnumber=KDC101>`__
driving a `PRM1/MZ8 (PRM1-Z8) <https://www.thorlabs.com/thorproduct.cfm?partnumber=PRM1/MZ8>`__
motorised rotation mount, using the `puzzlepiece <https://puzzlepiece.readthedocs.io>`__ framework.
Positions are in degrees.

The Piece has the same interface as :mod:`pzp_hardware.thorlabs.apt_stage`
(``serial``, ``connected``, ``pos``, Home, Identify), so it can be used in its place,
for example as ``stage:pos`` in a :mod:`pzp_hardware.generic.control.proportional` Piece.

The KDC101 is a Kinesis device, so this talks to the Kinesis C DLL
(``Thorlabs.MotionControl.KCube.DCServo.dll``) through ctypes rather than the APT server.

Example usage (see :ref:`getting-started` for more details on using Pieces in general)::

    import puzzlepiece as pzp
    from pzp_hardware.thorlabs import apt_rot

    app = pzp.QApp()
    puzzle = pzp.Puzzle(debug=False)
    puzzle.add_piece("rotator", apt_rot.Piece, row=0, column=0)
    puzzle.show()
    app.exec()

Installation
------------
* Install Kinesis from https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10285
  (usually ``C:\Program Files\Thorlabs\Kinesis``). If it is somewhere else, you will be asked
  for the directory the first time the Piece is used.
* Close the Kinesis app before connecting here - only one program can hold the device at a time.

Available Pieces
----------------
"""
import ctypes as c
import os
import time

import puzzlepiece as pzp
from puzzlepiece.extras import hardware_tools as pht

_DEFAULT_DLL_DIRECTORY = r"C:\Program Files\Thorlabs\Kinesis"
_DLL_NAME = "Thorlabs.MotionControl.KCube.DCServo.dll"
# Kinesis device type ID for the KDC101 (serial numbers start with 27)
_KDC101_TYPE = 27

# Status bits (CC_GetStatusBits)
_MOVING = 0x00000010 | 0x00000020 | 0x00000040 | 0x00000080
_HOMING = 0x00000200
_HOMED = 0x00000400
_ENABLED = 0x80000000

# Message queue (CC_GetNextMessage) types and IDs
_GENERIC_MOTOR = 2
_HOMED_MESSAGE = 0
_MOVED_MESSAGE = 1
_STOPPED_MESSAGE = 2
# Sent with every status poll
_STATUS_UPDATE = 3


def _load_kinesis():
    directory = _DEFAULT_DLL_DIRECTORY
    if not os.path.exists(os.path.join(directory, _DLL_NAME)):
        directory = pht.config(
            "kinesis_dll_directory",
            default=_DEFAULT_DLL_DIRECTORY,
            description="the Kinesis install directory (contains " + _DLL_NAME + ")",
            validator=pht.validator_path_exists,
        )
    pht.add_dll_directory(directory)
    lib = c.CDLL(os.path.join(directory, _DLL_NAME))

    serial = c.c_char_p
    for name, restype, argtypes in (
        ("TLI_BuildDeviceList", c.c_short, []),
        ("TLI_GetDeviceListByTypeExt", c.c_short, [c.c_char_p, c.c_ulong, c.c_int]),
        ("CC_Open", c.c_short, [serial]),
        ("CC_Close", None, [serial]),
        ("CC_Identify", None, [serial]),
        ("CC_EnableChannel", c.c_short, [serial]),
        ("CC_StartPolling", c.c_bool, [serial, c.c_int]),
        ("CC_StopPolling", None, [serial]),
        ("CC_GetPosition", c.c_int, [serial]),
        ("CC_MoveToPosition", c.c_short, [serial, c.c_int]),
        ("CC_Home", c.c_short, [serial]),
        ("CC_StopProfiled", c.c_short, [serial]),
        ("CC_GetStatusBits", c.c_uint32, [serial]),
        ("CC_RequestVelParams", c.c_short, [serial]),
        ("CC_GetVelParams", c.c_short, [serial, c.POINTER(c.c_int), c.POINTER(c.c_int)]),
        ("CC_SetVelParams", c.c_short, [serial, c.c_int, c.c_int]),
        ("CC_ClearMessageQueue", None, [serial]),
        ("CC_GetNextMessage", c.c_bool,
         [serial, c.POINTER(c.c_ushort), c.POINTER(c.c_ushort), c.POINTER(c.c_ulong)]),
    ):
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    return lib


#MARK: Piece
class Piece(pzp.Piece):
    """
    Main Piece for controlling a KDC101 + PRM1-Z8 rotation mount.

    ``pos`` and ``velocity`` are in degrees and degrees per second. ``homed`` reports
    whether the stage has been homed since it was powered on. Stop (or the Puzzle's
    global stop) interrupts a move or homing in progress.

    .. image:: ../images/pzp_hardware.thorlabs.apt_rot.Piece.png
    """
    # PRM1-Z8: 512 encoder counts/rev x 67:1 gearbox / 17.87 deg per output rev
    COUNTS_PER_UNIT = 512 * 67 / 17.87
    # KDC101 velocity and acceleration scaling for this stage (from the Kinesis docs)
    VELOCITY_SCALE = 42941.66
    ACCELERATION_SCALE = 14.66
    MAX_VELOCITY = 25.
    UNIT = "deg"

    def define_params(self):
        @pzp.param.dropdown(self, "serial", "27263761")
        def get_serials():
            if not self.puzzle.debug:
                return self._list_serials()

        @pzp.param.connect(self)
        def connect():
            if not self.puzzle.debug:
                serial = self._serial_bytes()
                self._check(self.lib.CC_Open(serial), "connect to")
                self._open_serial = serial
            try:
                if not self.puzzle.debug:
                    self.lib.CC_StartPolling(serial, 100)
                    self.lib.CC_ClearMessageQueue(serial)
                    self._check(self.lib.CC_EnableChannel(serial), "enable")
                    # Wait for the first status update so position and homed are valid
                    time.sleep(0.25)
                for name in ("pos", "velocity", "homed"):
                    self[name].get_value()
            except Exception:
                self._close()
                raise
            return 1

        @pzp.param.disconnect(self)
        def disconnect():
            if not self.puzzle.debug:
                self._close()
            return 0

        @pzp.param.spinbox(self, "pos", 0.)
        @self._ensure
        def move_to(value):
            if self.puzzle.debug:
                return value
            serial = self._open_serial
            self.lib.CC_ClearMessageQueue(serial)
            self._check(self.lib.CC_MoveToPosition(serial, self._to_counts(value)), "move")
            distance = abs(value - self._from_counts(self.lib.CC_GetPosition(serial)))
            self._wait_for((_MOVED_MESSAGE, _STOPPED_MESSAGE), 10 + 2 * distance / self._velocity())

        @move_to.set_getter(self)
        @self._ensure
        def get_position():
            if self.puzzle.debug:
                return self["pos"].value or 0.
            return self._from_counts(self.lib.CC_GetPosition(self._open_serial))

        @pzp.param.spinbox(self, "velocity", 10., v_min=0.1, v_max=self.MAX_VELOCITY)
        @self._ensure
        def set_velocity(value):
            # v_min/v_max only limit the GUI spinbox, not scripted set_value calls
            value = min(max(value, 0.1), self.MAX_VELOCITY)
            if self.puzzle.debug:
                return value
            acceleration, _ = self._get_vel_params()
            self._check(self.lib.CC_SetVelParams(
                self._open_serial, acceleration, int(round(value * self.VELOCITY_SCALE))
            ), "set the velocity of")

        @set_velocity.set_getter(self)
        @self._ensure
        def get_velocity():
            if self.puzzle.debug:
                return self["velocity"].value or 10.
            return round(self._velocity(), 3)

        @pzp.param.readout(self, "homed")
        @self._ensure
        def homed():
            if self.puzzle.debug:
                return True
            return bool(self.lib.CC_GetStatusBits(self._open_serial) & _HOMED)

    def define_actions(self):
        @pzp.action.define(self, "Home")
        @self._ensure
        def home():
            if not self.puzzle.debug:
                serial = self._open_serial
                self.lib.CC_ClearMessageQueue(serial)
                self._check(self.lib.CC_Home(serial), "home")
                # The PRM1-Z8 can go most of a turn to find its home switch
                self._wait_for((_HOMED_MESSAGE, _STOPPED_MESSAGE), 10 + 400 / self._velocity())
            else:
                self["pos"].set_value(0)
            self["pos"].get_value()
            self["homed"].get_value()

        @pzp.action.define(self, "Stop")
        def stop():
            self.call_stop()

        @pzp.action.define(self, "Identify")
        @self._ensure
        def identify():
            if self.puzzle.debug:
                return
            self.lib.CC_Identify(self._open_serial)

    #MARK: Helpers
    def _to_counts(self, value):
        return int(round(value * self.COUNTS_PER_UNIT))

    def _from_counts(self, counts):
        return round(counts / self.COUNTS_PER_UNIT, 4)

    def _serial_bytes(self):
        serial = str(self["serial"].value).strip()
        if not serial.isdigit():
            raise Exception(f"KDC101 serial number '{serial}' is not valid")
        return serial.encode()

    def _list_serials(self):
        self.lib.TLI_BuildDeviceList()
        buffer = c.create_string_buffer(1024)
        self.lib.TLI_GetDeviceListByTypeExt(buffer, len(buffer), _KDC101_TYPE)
        return [s for s in buffer.value.decode().split(",") if s]

    def _check(self, err_code, action):
        if err_code != 0:
            raise Exception(f"Failed to {action} KDC101 {self['serial'].value} (Kinesis error {err_code})")

    def _get_vel_params(self):
        acceleration, velocity = c.c_int(), c.c_int()
        self._check(self.lib.CC_GetVelParams(
            self._open_serial, c.byref(acceleration), c.byref(velocity)
        ), "read the velocity of")
        return acceleration.value, velocity.value

    def _velocity(self):
        return self._get_vel_params()[1] / self.VELOCITY_SCALE

    def _wait_for(self, message_ids, timeout):
        # Block until the KDC101 reports the move finished, keeping the GUI responsive
        # so the Stop action (or the Puzzle's global stop) can interrupt the move
        serial = self._open_serial
        self.stop = False
        message_type, message_id, data = c.c_ushort(), c.c_ushort(), c.c_ulong()
        end = time.time() + timeout
        finished = None
        while True:
            while self.lib.CC_GetNextMessage(serial, c.byref(message_type), c.byref(message_id), c.byref(data)):
                if message_type.value == _GENERIC_MOTOR and message_id.value in message_ids:
                    finished = time.time()
                elif finished and message_type.value == _STATUS_UPDATE:
                    # Position and status bits lag the "moved" message by one polling cycle
                    return
            if finished:
                if time.time() > finished + 1:
                    return
            elif time.time() > end:
                self.lib.CC_StopProfiled(serial)
                raise Exception(f"KDC101 {self['serial'].value} move timed out after {timeout:.0f} s")
            if self.stop:
                # call_stop has already stopped the motor; wait for it to settle
                while self.lib.CC_GetStatusBits(serial) & (_MOVING | _HOMING) and time.time() < end:
                    time.sleep(0.05)
                return
            self.puzzle.process_events()
            time.sleep(0.02)

    def _close(self):
        serial = getattr(self, "_open_serial", None)
        if serial is not None:
            self._open_serial = None
            self.lib.CC_StopPolling(serial)
            self.lib.CC_Close(serial)

    @pzp.piece.ensurer
    def _ensure(self):
        if self.puzzle.debug:
            return
        if getattr(self, "_open_serial", None) is None:
            raise Exception("KDC101 not connected")

    def call_stop(self):
        super().call_stop()
        if not self.puzzle.debug and getattr(self, "_open_serial", None) is not None:
            self.lib.CC_StopProfiled(self._open_serial)

    def setup(self):
        # The DLL is not needed (or loaded) in debug mode
        if self.puzzle.debug:
            return
        if not self.puzzle.globals.require("kinesis_kdc101"):
            self.puzzle.globals["kinesis_kdc101"] = _load_kinesis()
        self.lib = self.puzzle.globals["kinesis_kdc101"]

    def handle_close(self, event=None):
        super().handle_close(event)
        if not self.puzzle.debug and self.puzzle.globals.release("kinesis_kdc101"):
            del self.puzzle.globals["kinesis_kdc101"]


if __name__ == "__main__":
    app = pzp.QApp()
    puzzle = pzp.Puzzle(name="Rotation mount", debug=pht.debug_prompt())
    puzzle.add_piece("rotator", Piece, 0, 0)
    puzzle.show()
    app.exec()
