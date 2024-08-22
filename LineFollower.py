import numpy as np
import pygame
import time
import cv2
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder


# Load the KV file
Builder.load_string(
    """
<RobotController>:
    cam_display: cam_display

    Image:
        id: cam_display
        size: self.parent.size
        pos: self.parent.pos
"""
)

# PWM configuration
FREQ = 320  # servo frequency
DURATION = 1 / FREQ  # sample duration in s
SAMPLE_RATE = 40000  # sampling frequency
# user correction of the difference in motor speed forward/backward
CALIBRATION_LEFT = 0
CALIBRATION_RIGHT = -3
M1_SPEED_CALIBR = 0
M2_SPEED_CALIBR = 0
# user correction of the USB_C converter (+-100%)
DML = 0
DMR = 0

# PID configuration
P_VAL = 4
I_VAL = 0.01
D_VAL = 15
DEFAULT_SPEED = 12
MAX_SPEED = 70

# CAMERA configuration
FLASH_MODE = 1  # 0 - off, 1 - on

# ROBOT configuration
BLACK_VAL = 4000000
WHITE_VAL = 2000000
JUNCTION_VAL = 20000000
TURN_TIME = 1.5
OUT_OF_LINE_SPEED = 25
USE_NO_SHADOWS = False


class MotorController:
    def __init__(
        self,
        frequency=FREQ,
        duration=DURATION,
        sample_rate=SAMPLE_RATE,
        dml=DML,
        dmr=DMR,
        calibr_left=CALIBRATION_LEFT,
        calibr_right=CALIBRATION_RIGHT,
    ):
        self._frequency = frequency
        self._duration = duration
        self._sample_rate = sample_rate
        self._dml = dml
        self._dmr = dmr
        self._signal = np.int16(np.zeros((int(self._sample_rate * self._duration) * 2)))
        self._ml0 = 200
        self._mr0 = 100
        self._neutral_servo_val = 60
        self._calibr_left = calibr_left
        self._calibr_right = calibr_right

        # init sound
        pygame.mixer.pre_init(frequency=self._sample_rate, channels=2, allowedchanges=1)
        pygame.mixer.init()
        self._sound = pygame.mixer.Sound(self._signal.tobytes())

    def set(self, motor_left, motor_right):
        motor_left, motor_right = -motor_right, -motor_left
        # normailize values to range -10 to 10
        # to flip the servo direction remove the minus sign in front of int
        motor_left, motor_right = -int((motor_left + self._dml) / 5), -int(
            (motor_right + self._dmr) / 5
        )

        motor_left = self._set_constraints(-20, 20, motor_left)
        motor_right = self._set_constraints(-20, 20, motor_right)

        # multiples of 0.025ms = 25us (mikroseconds)
        # 40 - left   60 - stop   80 - right (1ms/1.5ms/2ms)
        if motor_left != self._ml0 or motor_right != self._mr0:
            duty_left = int(self._neutral_servo_val + self._calibr_left + motor_left)
            duty_right = int(self._neutral_servo_val + self._calibr_right - motor_right)

            # Generate signal for left channel
            self._signal[0 : (2 * duty_left) : 2] = -32767
            self._signal[(2 * duty_left) :: 2] = 32767

            # Generate signal for right channel
            self._signal[1 : (2 * duty_right + 1) : 2] = -32767
            self._signal[(2 * duty_right + 1) :: 2] = 32767

            self._sound.stop()
            self._sound = pygame.mixer.Sound(self._signal.tobytes())
            self._sound.play(-1)

            self._ml0, self._mr0 = motor_left, motor_right

    def _set_constraints(self, min_val, max_val, value):
        if value < min_val:
            return min_val
        if value > max_val:
            return max_val
        return value


class PIDRegulator:
    def __init__(
        self,
        p_val=P_VAL,
        i_val=I_VAL,
        d_val=D_VAL,
    ) -> None:
        self._p_val = p_val
        self._i_val = i_val
        self._d_val = d_val
        self._i_suma = 0
        self._err_old = 0

    def pid(self, error: float) -> int:
        p_index = error * self._p_val
        if self._i_suma * error < 0:
            self._i_suma = 0
        else:
            self._i_suma += error
        i_index = self._i_suma * self._i_val
        d_index = self._d_val * (error - self._err_old)
        self._err_old = error
        pid = int(p_index + i_index + d_index)
        return pid

    def get_err_old(self):
        return self._err_old


class CameraController:

    def __init__(self, image_widget, flash_mode=FLASH_MODE) -> None:
        self._image_widget = image_widget
        self._line_detect_arr = np.zeros((8))
        self._cap = cv2.VideoCapture(0)
        self._flash_mode = flash_mode
        self._cap.set(cv2.CAP_PROP_ANDROID_FLASH_MODE, self._flash_mode)

        if not self._cap.isOpened():
            self._image_widget.source = "error.png"
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self._image_arr = None
        self._text_to_print = []

    def refresh(self, dt):
        return_status, frame = self._cap.read()
        while not return_status:
            return_status, frame = self._cap.read()

        cv_image = self._process_frame(frame)
        self._detect_lines(cv_image)
        self._display_frame()

    def _process_frame(self, frame):
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width, height = self._image_widget.width, self._image_widget.height
        cv_width, cv_height = cv_image.shape[1], cv_image.shape[0]

        if (width > height) != (cv_width > cv_height):
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv_width, cv_height = cv_height, cv_width

        width = min(cv_width * height / cv_height, width)
        height = min(cv_height * width / cv_width, height)
        cv_image = cv2.resize(
            cv_image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR
        )

        self._image_arr = np.array(cv_image)
        return cv_image

    def _detect_lines(self, cv_image):
        width_coordinate = self._image_arr.shape[1] // 8
        height_coordinate = self._image_arr.shape[0] // 2
        vertical_start = height_coordinate - width_coordinate // 2
        vertical_end = height_coordinate + width_coordinate // 2
        max_val = 255 * (vertical_end - vertical_start) * width_coordinate

        for i in range(8):
            cv2.rectangle(
                self._image_arr,
                (i * width_coordinate, vertical_start),
                ((i + 1) * width_coordinate, vertical_end),
                (0, 0, 0),
                2,
            )
            square_sum = max_val - np.sum(
                self._image_arr[
                    vertical_start:vertical_end,
                    i * width_coordinate : (i + 1) * width_coordinate,
                ]
            )
            self._line_detect_arr[i] = square_sum

        self.add_text(f"Line detect: {self._line_detect_arr}", (30, 50))

    def _display_frame(self):
        self._print_all_text()
        buf = cv2.flip(self._image_arr, 0).tobytes()
        texture = Texture.create(
            size=(self._image_arr.shape[1], self._image_arr.shape[0]),
            colorfmt="luminance",
        )
        texture.blit_buffer(buf, colorfmt="luminance", bufferfmt="ubyte")
        self._image_widget.texture = texture

    def get_line_detect_arr(self):
        return self._line_detect_arr

    def _print_text(self, text: str, coordinates: tuple):
        cv2.putText(
            self._image_arr,
            text,
            coordinates,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def _print_all_text(self):
        for text in self._text_to_print:
            self._print_text(text[0], text[1])
        self._text_to_print = []

    def add_text(self, text: str, coordinates: tuple):
        self._text_to_print.append((text, coordinates))


class RobotController(Widget):

    def __init__(
        self,
        white_val=WHITE_VAL,
        black_val=BLACK_VAL,
        out_of_line_speed=OUT_OF_LINE_SPEED,
        default_speed=DEFAULT_SPEED,
        max_speed=MAX_SPEED,
        use_no_shadows=USE_NO_SHADOWS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._motors_controller = MotorController()
        self._pid_regulator = PIDRegulator()
        self._camera_controller = CameraController(self.ids.cam_display)
        self._weights = self._set_weights()
        self._white_val = white_val
        self._black_val = black_val
        self._bw_border = (self._white_val + self._black_val) // 2
        self._out_of_line_speed = out_of_line_speed
        self._default_speed = default_speed
        self._max_speed = max_speed
        self._use_no_shadows = use_no_shadows
        self.junction = False
        self.turning = False
        self._turn_start_time = 0
        self._turn_time = TURN_TIME
        Clock.schedule_once(self.start, 1)

    def start(self, dt):
        self._camera_controller.refresh(dt)
        Clock.schedule_interval(self._camera_controller.refresh, 1.0 / 30)
        Clock.schedule_interval(self.refresh, 1.0 / 30)

    def refresh(self, dt):
        try:
            line_detect_arr = self._camera_controller.get_line_detect_arr()
            if self._use_no_shadows:
                line_detect_arr = self._eliminate_shadows(line_detect_arr)
                self._camera_controller.add_text(
                    f"No shadows: {line_detect_arr}", (30, 100)
                )
            self._set_motors(line_detect_arr)

        except Exception as e:
            pass

    def _eliminate_shadows(self, arr) -> None:
        return [
            self._black_val if x > self._bw_border else self._white_val for x in arr
        ]

    def _out_of_line(self, arr):
        if max(arr) < self._bw_border:
            return True
        return False

    def _detect_junction(self, arr):
        if sum(arr) > JUNCTION_VAL:
            self.junction = True
            self._camera_controller.add_text(
                f"Junction Detected: {sum(arr)}", (30, 300)
            )
        else:
            self.junction = False

    def _turn_90_deg_right(self):
        self._motors_controller.set(self._out_of_line_speed, -self._out_of_line_speed)

    def _back_to_line(self) -> float:
        err_old = self._pid_regulator.get_err_old()
        if err_old < 0:
            self._motors_controller.set(
                -self._out_of_line_speed, self._out_of_line_speed
            )
        else:
            self._motors_controller.set(
                self._out_of_line_speed, -self._out_of_line_speed
            )

    def _set_weights(self) -> list:
        weights = [0 for _ in range(8)]
        for i in range(4):
            weights[4 + i] = i + 1
            weights[i] = -4 + i
        return weights

    def _compute_err(self, arr) -> float:
        weighted_sum = 0
        for i in range(8):
            weighted_sum += arr[i] * self._weights[i]
        err = weighted_sum / sum(arr)
        return err

    def _set_motors(self, arr) -> None:
        self._detect_junction(arr)
        if self.junction:
            self._turn_start_time = time.time()
        if time.time() - self._turn_start_time < self._turn_time:
            self._turn_90_deg_right()
        else:
            if self._out_of_line(arr):
                self._back_to_line()
            else:
                err = self._compute_err(arr)
                self._camera_controller.add_text(f"Error: {err}", (30, 150))
                pid = self._pid_regulator.pid(err)
                self._camera_controller.add_text(f"PID: {pid}", (30, 200))
                m1 = self._default_speed + pid
                m2 = self._default_speed - pid
                m1 = self._set_constraints(m1)
                m2 = self._set_constraints(m2)
                self._motors_controller.set(m1, m2)
                self._camera_controller.add_text(f"Motors: {m1 ,m2}", (30, 250))

    def _set_constraints(self, value):
        if value < -self._max_speed:
            return -self._max_speed
        if value > self._max_speed:
            return self._max_speed
        return value


class LineFollower(App):

    def build(self):
        return RobotController()


if __name__ == "__main__":
    LineFollower().run()
