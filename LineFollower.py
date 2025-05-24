import numpy as np
import pygame
import cv2
import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pygame
from typing import Tuple
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
import threading
from queue import Queue
import queue

# TODO: Stereo not working properly

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
SAMPLE_RATE = 40000
SAMPLES = 250
DURATION = 1 / SAMPLES
INT16_MAX = np.iinfo(np.int16).max
STOP = 60
FRONT_RANGE = (55, 21)
BACK_RANGE = (62, 96)
PYGAME_DELAY = 1

# PID configuration
P_VAL = 10
I_VAL = 0.15
D_VAL = 10
DEFAULT_SPEED = 10
MAX_SPEED = 70

# CAMERA configuration
FLASH_MODE = 1  # 0 - off, 1 - on
RES_WIDTH = 1280
RES_HEIGHT = 720


# ROBOT configuration
BLACK_VAL = 4000000
WHITE_VAL = 2000000
OUT_OF_LINE_CONDITION = True
OUT_OF_LINE_SPEED = 15
USE_NO_SHADOWS = False


class PWM:

    def __init__(
        self,
        samples: int = SAMPLES,
        front_range: Tuple[int, int] = FRONT_RANGE,
        back_range: Tuple[int, int] = BACK_RANGE,
        stop: int = STOP,
        curve_len: int = 10,
    ):
        self._samples = samples
        self._int16_max = np.iinfo(np.int16).max
        self._front_range = front_range
        self._back_range = back_range
        self._stop = stop
        self._curve_len = curve_len

    def _speed_to_samples(self, speed: int) -> int:
        if speed == 0:
            return self._stop
        if speed > 0:
            return self._range_to_samples(speed, self._front_range)
        if speed < 0:
            return self._range_to_samples(-speed, self._back_range)

    def _range_to_samples(self, speed: int, sample_range, max_speed=100) -> int:
        if speed > max_speed:
            raise ValueError(f"Speed must be less than {max_speed}")

        return int(
            ((speed - 1) * (sample_range[1] - sample_range[0]) / (max_speed - 1))
            + sample_range[0]
        )

    def _signal_mono(self, samples: int, polarity: int = 1) -> npt.NDArray[np.int16]:
        self._validate_params(samples)
        print(f"Samples: {samples}")

        # length of the curve edge at the beginning and end of the signals
        arr_len = self._samples // 2
        signal_array = np.full(arr_len, polarity * self._int16_max, dtype=np.int16)
        amplitude_start = (arr_len - samples) // 2
        signal_array[0] = 0

        # rising edge of the signal
        start = 1
        for i in range(start, amplitude_start):
            signal_array[i] = (
                polarity * 2.5 / (6 * (amplitude_start - i)) * self._int16_max
            )

        # curved edge of the signal
        start = amplitude_start
        end = amplitude_start + self._curve_len
        for i in range(start, end):
            sinus = amplitude_start - 1
            angle = 1 + (i - sinus) * 6 / 100
            signal_array[i] = polarity * int(math.sin(angle) * self._int16_max)

        # copy the first half of the signal to the second half
        signal_array[arr_len - end : arr_len] = signal_array[0:end][::-1]
        return signal_array

    def _signal_stereo(
        self, samples1: int, samples2: int, polarity1: int, polarity2: int
    ) -> npt.NDArray[np.int16]:
        signal_array1 = self._signal_mono(samples1, polarity1)
        signal_array2 = self._signal_mono(samples2, polarity2)
        return np.vstack((signal_array1, signal_array2), dtype=np.int16)

    def _validate_params(self, samples: int) -> None:
        print(f"Samples: {samples}")
        if samples < 1:
            raise ValueError("Samples must be at least 1")
        if samples > self._samples // 2 - (2 * self._curve_len):
            raise ValueError(
                f"Samples must be less than {self._samples // 2 - (2 * self._curve_len)}"
            )

    def visualize_signal(self, signal: npt.NDArray[np.int16]):
        plt.figure(figsize=(8, 5))
        plt.xlabel("Osa X")
        plt.ylabel("Osa Y")
        plt.legend()
        plt.grid(True)
        colors = ["b", "g"]
        if signal.ndim > 1:
            for i in range(len(signal)):
                plt.plot(signal[i], linestyle="-", color=colors[i])
        else:
            plt.plot(signal, linestyle="-", color="b")
        plt.show()

    def get_signal(
        self, left: int, right: int, channels: int = 2
    ) -> npt.NDArray[np.int16]:
        if channels not in (1, 2):
            raise ValueError("Channels must be 1 or 2")

        samples1 = self._speed_to_samples(left)
        samples2 = self._speed_to_samples(right)

        if channels == 1:
            return np.concatenate(
                (self._signal_mono(samples1, 1), self._signal_mono(samples2, -1))
            )

        if channels == 2:
            samples1 = self._speed_to_samples(left)
            samples2 = self._speed_to_samples(right)
            return self._signal_stereo(samples1, samples2, 1, -1)


class MotorController:
    def __init__(
        self,
        pwm: PWM,
        sample_rate: int = SAMPLE_RATE,
        channels: int = 2,
        max_speed: int = 100,
        pygame_delay: int = PYGAME_DELAY,
    ):
        self._pwm = pwm
        self._sample_rate = sample_rate
        self._channels = channels
        self._max_speed = max_speed
        self._pygame_delay = pygame_delay
        self._sound: pygame.mixer.Sound = None

        pygame.mixer.pre_init(frequency=self._sample_rate, channels=self._channels)
        pygame.mixer.init()

    def set(self, motor_left, motor_right):
        motor_right = -motor_right
        motor_left = max(-self._max_speed, min(motor_left, self._max_speed))
        motor_right = max(-self._max_speed, min(motor_right, self._max_speed))
        signal = self._pwm.get_signal(motor_left, motor_right, self._channels)
        if self._sound:
            self._sound.stop()
        self._sound = pygame.mixer.Sound(signal.tobytes())
        self._sound.play(-1)
        pygame.time.delay(self._pygame_delay)


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
        self.frame_queue = Queue(maxsize=1)
        self._cap = cv2.VideoCapture(0, cv2.CAP_ANDROID)
        self._flash_mode = flash_mode
        self._cap.set(cv2.CAP_PROP_ANDROID_FLASH_MODE, self._flash_mode)

        if not self._cap.isOpened():
            self._image_widget.source = "error.png"
        else:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, 60)
            threading.Thread(target=self._reader_thread, daemon=True).start()

        self._image_arr = None
        self._text_to_print = []

    def _reader_thread(self):
        while True:
            ret, frame = self._cap.read()
            if ret:
                if self.frame_queue.full():
                    _ = self.frame_queue.get_nowait()
                self.frame_queue.put(frame)

    def refresh(self, dt):
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            return
        cv_image = self._process_frame(frame)

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
        self._pwm = PWM()
        self._motors_controller = MotorController(self._pwm, channels=1)
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

    def _back_to_line(self) -> float:
        err_old = self._pid_regulator.get_err_old()
        if err_old > 0:
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
        if self._out_of_line(arr) and OUT_OF_LINE_CONDITION:
            self._back_to_line()
        else:
            err = self._compute_err(arr)
            self._camera_controller.add_text(f"Error: {err}", (30, 150))
            pid = self._pid_regulator.pid(err)
            self._camera_controller.add_text(f"PID: {pid}", (30, 200))
            m_right = self._default_speed - pid
            m_left = self._default_speed + pid
            self._motors_controller.set(m_right, m_left)
            self._camera_controller.add_text(
                f"Motors: L:{m_left} | R:{m_right}", (30, 250)
            )


class LineFollower(App):

    def build(self):
        return RobotController()


if __name__ == "__main__":
    LineFollower().run()
