function pc = pcfrom3dgs(gaussians, colorInfo)
% colorInfo : {"RGB", "Opacity"}

if nargin < 2
    colorInfo = "RGB";
end

v = [gaussians.x, gaussians.y, gaussians.z];
op = 1 ./ (1 + exp(-gaussians.opacity));

if strcmp(colorInfo, "RGB")
    SH_C0 = 0.28209479177387814;
    
    f_dc = [gaussians.f_dc_0, gaussians.f_dc_1, gaussians.f_dc_2];
    clr = (f_dc * SH_C0) + 0.5;
    clr(clr < 0) = 0;
    clr(clr > 1) = 1;
elseif strcmp(colorInfo, "Opacity")
    cm = jet(256);
    x = linspace(0, 1, 256);
    clr = interp1(x, cm, op);
else
    assert(0);    
end

pc = pointCloud( v, "Color", clr, "Intensity", op );