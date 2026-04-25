function gsshow(gaussians, axisflag)

if nargin < 2, axisflag = true; end

pc = pcfrom3dgs(gaussians);
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

