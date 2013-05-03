from pyplasm import *

# Create a cube of size 2:
c = CUBE(2)
# Create a cylinder of radius 0.75 and height 4:
cyl = CYLINDER(0.75, 4)
# Translate the cube by -1 in each axial direction:
c = T(c, -1, -1, -1)
# Translate the cylinder by -2 in the z-direction:
cyl = T(cyl, 0, 0, -2)
# Rotate the cylinder by 90 degrees about the x-axis:
cyl2 = RDEG(cyl, 1, 90)
# Rotate the cylinder by 90 degrees about the z-axis:
cyl3 = RDEG(cyl2, 3, 90)
# Subtract the cylinder from the cube:
c = DIFF(c, cyl, cyl2, cyl3)

col = [0, 0, 255, 0]

c = COLOR(c, col)

# View the result. 
VIEW(c)

# STL output:
#import plasm_stl
#filename = "drilled-cube.stl"
#plasm_stl.toSTL(c, filename)
#print "STL file written to", filename
