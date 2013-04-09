from pyplasm import *

c = CUBE(1)

color1 = [0, 0, 1, 1]
color2 = [0, 0, 1, 0.1]

c1 = COLOR(color2)(c)
c2 = COLOR(color2)(c)

VIEW(c1)
VIEW(c2)
