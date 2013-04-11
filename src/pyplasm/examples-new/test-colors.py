from pyplasm import *


c = CUBE(2)
d = CUBE(1)
d = T(d, -0.1, -0.1, -0.1)

c = COLOR(c, [255, 0, 0])
d = COLOR(d, [0, 0, 255])

#e = STRUCT(c, d)
e = XOR(c, d)

VIEW(e)
