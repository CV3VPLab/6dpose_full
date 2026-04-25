function gaussians = read_3dgs_ply(fn)

data = plyread(fn);
gaussians = data.vertex;