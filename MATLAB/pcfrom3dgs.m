function pc = pcfrom3dgs(gaussians)

SH_C0 = 0.28209479177387814;

f_dc = [gaussians.f_dc_0, gaussians.f_dc_1, gaussians.f_dc_2];
rgb = (f_dc * SH_C0) + 0.5;
rgb(rgb < 0) = 0;
rgb(rgb > 1) = 1;

v = [gaussians.x, gaussians.y, gaussians.z];
n = [gaussians.nx, gaussians.ny, gaussians.nz];
pc = pointCloud( v, "Color", rgb, "Normal", n, "Intensity", gaussians.opacity );