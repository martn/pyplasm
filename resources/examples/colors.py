from pyplasm import *

def out():
    point=MKPOL([[[0]],[[1]],[[1]]])
    edge=CUBOID([1])
    face=CUBOID([1,1])
    cell=CUBOID([1,1,1])

    a = STRUCT(
            COLOR(RED)(point)
            ,COLOR(GREEN)(T(point, 1, 0, 0))
            ,COLOR(BLUE)(T(point, 2, 0, 0))
            ,COLOR(RED)(T(edge, 3, 0, 0))
            ,COLOR(GREEN)(T(edge, 4, 0, 0))
            ,COLOR(BLUE)(T(edge, 5, 0, 0))
            ,COLOR(RED)(T(face, 6, 0, 0))
            ,COLOR(GREEN)(T(face, 7, 0, 0))
            ,COLOR(BLUE)(T(face, 8, 0, 0))
            ,COLOR(RED)(T(cell, 9, 0, 0))
            ,COLOR(GREEN)(T(cell, 10, 0, 0))
            ,COLOR(BLUE)(T(cell, 11, 0, 0)))

    return a

VIEW(out())

