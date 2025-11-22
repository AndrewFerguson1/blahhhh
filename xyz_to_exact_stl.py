import sys, numpy as np
from pathlib import Path
from scipy.interpolate import griddata

def build_grid_mesh(GX, GY, GZ):
    ny, nx = GX.shape
    V = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
    F = []
    def idx(i,j): return i*nx + j
    for i in range(ny-1):
        for j in range(nx-1):
            v00 = idx(i,j); v10 = idx(i+1,j); v01 = idx(i,j+1); v11 = idx(i+1,j+1)
            F.append([v00, v10, v11])
            F.append([v00, v11, v01])
    return V, np.asarray(F, dtype=np.int32)

def write_ascii_stl(path, V, F, name="seismic_exact_scale"):
    def nrm(a,b,c):
        n = np.cross(b-a, c-a); L = np.linalg.norm(n)
        return n/L if L>0 else np.array([0,0,0])
    with open(path, "w") as f:
        f.write(f"solid {name}\n")
        for a,b,c in F:
            v0,v1,v2 = V[a],V[b],V[c]
            nx,ny,nz = nrm(v0,v1,v2)
            f.write(f"  facet normal {nx} {ny} {nz}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
            f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
            f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")

def write_obj(path, V, F):
    with open(path, "w") as f:
        f.write("# OBJ exact-scale seismic surface\n")
        for v in V: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for a,b,c in F: f.write(f"f {a+1} {b+1} {c+1}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python xyz_to_exact_stl.py seismic.txt")
        sys.exit(1)
    in_path = Path(sys.argv[1])
    pts = np.loadtxt(in_path)  # whitespace (tabs/spaces) OK
    X, Y, Z = pts[:,0], pts[:,1], pts[:,2]

    # ===== Regular grid (no scaling, no exaggeration) =====
    # pick resolution appropriate for ~123 points:
    nx, ny = 60, 60
    gx = np.linspace(X.min(), X.max(), nx)
    gy = np.linspace(Y.min(), Y.max(), ny)
    GX, GY = np.meshgrid(gx, gy)

    Zi_lin = griddata((X, Y), Z, (GX, GY), method="linear")
    Zi_near = griddata((X, Y), Z, (GX, GY), method="nearest")
    Zi = np.where(np.isnan(Zi_lin), Zi_near, Zi_lin)

    Vsurf, Fsurf = build_grid_mesh(GX, GY, Zi)

    # ===== Close it to a solid at z = min(Z) (exact scale preserved) =====
    zmin = float(Z.min()); xmin, xmax = float(X.min()), float(X.max())
    ymin, ymax = float(Y.min()), float(Y.max())

    # Add base rectangle (z = zmin)
    base_rect = np.array([[xmin,ymin,zmin],
                          [xmax,ymin,zmin],
                          [xmax,ymax,zmin],
                          [xmin,ymax,zmin]], float)
    n_surface = Vsurf.shape[0]
    V = np.vstack([Vsurf, base_rect])

    # Base faces (two tris)
    i0,i1,i2,i3 = n_surface+0, n_surface+1, n_surface+2, n_surface+3
    F = []
    F.extend(Fsurf.tolist())
    F += [[i0,i1,i2],[i0,i2,i3]]

    # Perimeter ring of surface grid for walls
    def idx(i,j): return i*nx + j
    perim = []
    for j in range(nx): perim.append(idx(0,j))
    for i in range(1,ny): perim.append(idx(i,nx-1))
    for j in range(nx-2,-1,-1): perim.append(idx(ny-1,j))
    for i in range(ny-2,0,-1): perim.append(idx(i,0))

    # Snap each perimeter XY to nearest base rectangle edge at z=zmin
    def edge_snap(x,y):
        dx0, dx1 = abs(x-xmin), abs(x-xmax)
        dy0, dy1 = abs(y-ymin), abs(y-ymax)
        m = min(dx0,dx1,dy0,dy1)
        if m == dx0:   return [xmin, y, zmin]
        elif m == dx1: return [xmax, y, zmin]
        elif m == dy0: return [x, ymin, zmin]
        else:          return [x, ymax, zmin]

    base_ring = np.array([edge_snap(*Vsurf[v][:2]) for v in perim], float)
    idx_ring = V.shape[0]
    V = np.vstack([V, base_ring])

    # Side walls (quads split into tris)
    for k in range(len(perim)):
        ta = perim[k]; tb = perim[(k+1)%len(perim)]
        ba = idx_ring + k; bb = idx_ring + (k+1)%len(perim)
        F += [[ta,tb,bb],[ta,bb,ba]]

    F = np.array(F, dtype=np.int32)

    # ===== Write files =====
    out_stl = in_path.with_name("seismic_exact_scale.stl")
    out_obj = in_path.with_name("seismic_exact_scale.obj")
    write_ascii_stl(out_stl, V, F)
    write_obj(out_obj, V, F)
    print(f"OK: {out_stl}  |  {out_obj}")
    print(f"Vertices: {V.shape[0]}  Faces: {F.shape[0]}  zmin={zmin}")

if __name__ == "__main__":
    main()