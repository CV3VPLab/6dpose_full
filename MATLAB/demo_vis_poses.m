%%
clear all
close all

fn_model = '..//data//object//can_mtdew//model//point_cloud.ply';
fn_poses = '..//data//output//can_mtdew//result//refined_pose.json';

gaussians = read_3dgs_ply(fn_model);

% pcModel = pcfrom3dgs(gaussians);
pcModel = pcread("pc_can_mtdew.ply");

[T, loss] = poseEstimAnal(fn_poses);

figure, hold on, grid on, xyzlabel;
axis vis3d

ax3 = diag([0.05, 0.05, 0.09]);
org_obj = zeros(length(T), 3);
x_obj = zeros(length(T), 3);
y_obj = zeros(length(T), 3);
z_obj = zeros(length(T), 3);
for i = 1:length(T)
    R = rotvec2mat3d(rotmat2vec3d(T{i,1}));
    tform = rigidtform3d( R, T{i,2} );
    
    % Transform the point cloud model using the estimated transformation
    transformedPcModel = pctransform(pcModel, tform);
    pcshow(transformedPcModel)
    ax31 = R * ax3;
    org_obj(i, :) = T{i,2}';
    x_obj(i, :) = ax31(:, 1)';
    y_obj(i, :) = ax31(:, 2)';
    z_obj(i, :) = ax31(:, 3)';
end

quiver3(org_obj(:,1), org_obj(:,2), org_obj(:,3), ...
    x_obj(:,1), x_obj(:,2), x_obj(:,3), 0, 'r', 'LineWidth', 1)
quiver3(org_obj(:,1), org_obj(:,2), org_obj(:,3), ...
    y_obj(:,1), y_obj(:,2), y_obj(:,3), 0, 'g', 'LineWidth', 1)
quiver3(org_obj(:,1), org_obj(:,2), org_obj(:,3), ...
    z_obj(:,1), z_obj(:,2), z_obj(:,3), 0, 'b', 'LineWidth', 1)

xlim([-0.5, 0.5]);
ylim([-0.2, 0.2]);
zlim([0, 0.9]);

campos([0.0382   -1.4586   -6.4145])