function gsshow(gaussians, colorInfo, axisflag)

if nargin < 2, colorInfo = "RGB"; end
if nargin < 3, axisflag = true; end

pc = pcfrom3dgs(gaussians, colorInfo);
pcshow(pc)
xlabel('x'), ylabel('y'), zlabel('z')

if axisflag
    hold on
    v = pc.Location;
    d = max(vecnorm(v'));

    line([-d, d], [0, 0], [0, 0], "Color", "r", "LineWidth", 1);
    line([0, 0], [-d, d], [0, 0], "Color", "g", "LineWidth", 1);
    line([0, 0], [0, 0], [-d, d], "Color", "b", "LineWidth", 1);
    hold off
end

if strcmp(colorInfo, "Opacity")
    colormap('jet');
    colorbar
end

