import numpy as np

def distance(a, b):
    return float(np.linalg.norm(np.array(a.coords) - np.array(b.coords)))

def dihedral(a, b, c, d):
    p0, p1, p2, p3 = (np.array(x.coords) for x in (a, b, c, d))
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2/np.linalg.norm(b2))
    return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))

def angle(a, b, c):
    p0, p1, p2 = (np.array(x.coords) for x in (a, b, c))
    v1, v2 = p0 - p1, p2 - p1
    return float(np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))

