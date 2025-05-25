# Line Folower
Python program for a line following robot driven by a PID regulation. The code is meant to by run on android device using Pydroid.
## How to build the robot
[tutorial](https://bitbeam4.eu/index.php/stavebni-navody/roboti/robot-pro-experty/)

## Configuration (to be added)

### PWM configuration
```
FREQ = 320  # servo frequency
DURATION = 1 / FREQ  # sample duration in s
SAMPLE_RATE = 40000  # sampling frequency
```
```

### PID configuration
```
P_VAL = 6
I_VAL = 0
D_VAL = 0
```

### ROBOT configuration
```
BLACK_VAL = 4000000
WHITE_VAL = 2000000
DEFAULT_SPEED = 10
MAX_SPEED = 50
OUT_OF_LINE_SPEED = 25
USE_NO_SHADOWS = False
```

