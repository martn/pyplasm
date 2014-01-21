from pyplasm import *
from scipy import reshape

def cumsum(iterable):
    """ Cumulative addition: list(cumsum(range(4))) => [0, 1, 3, 6] 
        
        Return a list of numbers
    """
    iterable = iter(iterable)
    s = iterable.next()
    yield s
    for c in iterable:
        s = s + c
        yield s

def larExtrude(model,pattern):
    """ Multidimensional extrusion 
        model is a LAR model: a pair (vertices, cells)
        pattern is a list of positive and negative sizes (multi-extrusion)
        
        Return a "model"
    """
    V, FV = model
    d, m = len(FV[0]), len(pattern)
    coords = list(cumsum([0]+(AA(ABS)(pattern))))
    offset, outcells, rangelimit = len(V), [], d*m
    for cell in FV:
        tube = [v + k*offset for k in range(m+1) for v in cell]
        cellTube = [tube[k:k+d+1] for k in range(rangelimit)]
        outcells += [reshape(cellTube, newshape=(m,d,d+1)).tolist()]
    
    outcells = AA(CAT)(TRANS(outcells))
    cellGroups = [group for k,group in enumerate(outcells) if pattern[k]>0 ]
    outVertices = [v+[z] for z in coords for v in V]
    outModel = outVertices, CAT(cellGroups)
    return outModel

def larSimplexGrid(shape):
    """ User interface in LARCC.
        
        Return an (hyper-)cuboid of given shape. Vertices have integer coords
    """
    model = V0,CV0 = [[]],[[0]]    # the empty simplicial model
    for item in shape:
        model = larExtrude(model,item * [1])
    return model


def SIMPLEXGRID(size):
    """ User interface in Pyplasm.
        size = list of grid sizes in each coordinate direction;
        shape = list of numbers of steps in each coordinate direction.
        
        SIMPLEXGRID(size)(shape): Return an HPC value
    """
    def model2hpc0(shape):
        assert len(shape) == len(size)
        scaleCoeffs = map(DIV, zip(size,shape))
        model = larSimplexGrid(shape)
        verts,cells = model
        cells = [[v+1 for v in cell] for cell in cells]
        coords = range(1,len(size)+1)
        return PLASM_S(coords)(scaleCoeffs)(MKPOL([verts,cells,None]))
    return model2hpc0
