from pyplasm import *

# Main panel:
c = BRICK(3, 2, 0.1)
c = T(c, 0, 0, -0.1)

# LCD screen
screen = BRICK(2.8, 1.8, 0.05)
screen = T(screen, 0.1, 0.1, -0.0499)
blue = [0, 0, 100]
screen = COLOR(screen, blue)

# Leg:
d = CYLINDER(0.1, 2)
d = S(d, 3, 1, 1)
d = R(d, 1, -PI/3)
d = T(d, 1.5, -0.5, -1.1)

# Base
d2 = CYLINDER(0.5, 0.1)
d2 = S(d2, 2, 1, 1)
d2 = R(d2, 1, -PI/3)
d2 = T(d2, 1.5, -0.5, -1.1)

# Frame:
frame = U(c, d, d2)
grey = [150, 150, 150]
screen = COLOR(screen, grey)

# Display it:
monitor = STRUCT(frame, screen)
VIEW(monitor)
