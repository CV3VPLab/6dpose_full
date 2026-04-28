%%
clear all
close all

fn = '..\data\can_data\3dgs_pepsi_pinset\pepsi_painted_canonical\point_cloud\iteration_30000\point_cloud.ply';

gaussians = read_3dgs_ply(fn);
figure, gsshow(gaussians), title("Point cloud of 3DGS")
figure, gsshow(gaussians, "Opacity"), title("Opacity map")

pcModel = pcfrom3dgs(gaussians);

%% 
pyenv(Version="C:\Users\choik\anaconda3\envs\gsplat\python.exe");

paths.data = '..\data\can_data';
paths.render = fullfile( paths.data, 'gallery_renders_gs_ds' );
paths.depth = fullfile( paths.data, 'gallery_depth_gs_ds' );
paths.xyz = fullfile( paths.data, 'gallery_xyz_gs_ds' );

depth = py.numpy.load( fullfile(paths.depth, "0000.npy") );
depth = single(depth);
rgb = imread( fullfile(paths.render, "0000.png") );
K = readmatrix( fullfile(paths.data, "intrinsics_ds.txt") );

poses_json = jsondecode(fileread(fullfile(paths.data, 'gallery_poses.json')));
poses = poses_json.poses;

%% Depth로 point cloud 구성
pc = pcfromDepthmap( depth, K, rgb );

% RGB 시각화
figure, pcshow( pc.Location ), title("Point cloud obtained from depth map (PCD)")
xlabel('x'), ylabel('y'), zlabel('z')

figure, pcshow( pc ), title("PCD with rendered color")
f = gcf;
f.Color = 'w';
ax = gca;
ax.Color = 'w';
xlabel('x'), ylabel('y'), zlabel('z')

%%
tform = rigidtform3d( poses(1).T_obj_to_cam );
X = transformPointsForward(tform, pcModel.Location );
pcModel2 = pointCloud( X, "Color", pcModel.Color );

figure, pcshow( pc.Location ), title("PCD & 3DGS of rendering pose")
xlabel('x'), ylabel('y'), zlabel('z')
hold on
pcshow( pcModel2 )
hold off



