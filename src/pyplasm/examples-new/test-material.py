from pyplasm import *

# Outer cylinder:
Rout = 0.10
Hout = 0.10
cyl_outer = CYLINDER(Rout, Hout)

# Inner cylinder
Rin = 0.08
bt = 0.005
cyl_inner = CYLINDER(Rin, Hout)
cyl_inner = T(cyl_inner, 0, 0, bt)

# Subtract inner cylinder from outer:
out = DIFF(cyl_outer, cyl_inner)



 
ambient = [1,0,0,1]
diffuse = [0,1,0,1]
specular = [0,0,1,0]
emission = [0,0,0,1]
shine = 100 

#out = MATERIAL([1,0,0,1,   0,1,0,1,   0,0,1,1,    0,0,0,1,    100])(out)
out = MATERIAL([0.1, 0.1, 0.1, 1.0,   0.1, 0.1, 0.8, 0.5,  0, 0, 0, 1.0,   0.0, 0.0, 0.0, 1.0,   10.0])(out)



# Display the result:

out = COLOR(out, [0, 255, 0, 0.0])

VIEW(out)

# STL output:
#import plasm_stl
#filename = "ashtray.stl"
#plasm_stl.toSTL(out, filename)
#print "STL file written to", filename


