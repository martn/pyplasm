from pyplasm import *

# Create a cube of size 2:
c = CUBE(2)
# Translate the cube by -1 in each axial direction:
c = T(c, -1, -1, -1)
# Create a cylinder of radius 0.75 and height 4:
cyl = CYLINDER(0.75, 4)
# Translate the cylinder by -2 in the z-direction:
cyl = T(cyl, 0, 0, -2)
cyl = COLOR(cyl, [255, 0, 0])
cyl2 = RDEG(cyl, 2, 90)
cyl2 = COLOR(cyl2, [0, 255, 0])
# Rotate the cylinder by 90 degrees about the x-axis:
#cyl2 = RDEG(cyl, 1, 90)
#cyl2 = COLOR(cyl2, [0, 255, 0])
# Rotate the cylinder by 90 degrees about the z-axis:
#cyl3 = RDEG(cyl2, 3, 90)
#cyl3 = COLOR(cyl3, [0, 0, 255])
# View the result. 
#VIEW(STRUCT(c, cyl, cyl2, cyl3))
VIEW(STRUCT(cyl, cyl2))

print RED
