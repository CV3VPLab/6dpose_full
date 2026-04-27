function pc = pcfromDepthmap( depth, K, rgb )
% depth : 2D depth map
% K : intrinsic matrix
% rgb : rendered image (optional)

[H,W] = size(depth, [1, 2]);
dValidIdx = depth > 1e-8;
[gx, gy] = meshgrid(0:(W-1), 0:(H-1));
gxv = gx(dValidIdx);
gyv = gy(dValidIdx);
dv = depth(dValidIdx);

p = [gxv, gyv, ones(length(gxv), 1)];
P = (p .* dv) / K'; % Camera 기준 3D points from depth map

if nargin == 3
    rgbv = reshape( rgb, [], 3 );
    rgbv = rgbv(dValidIdx, :);

    pc = pointCloud(P, "Color", rgbv);
else
    assert( nargin == 2 );
    pc = pointCloud(P);
end