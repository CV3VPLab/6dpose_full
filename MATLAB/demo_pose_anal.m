%%
clear all

poses_GS = cell(1,5);
poses_RR = cell(1,5);
poses_MX = cell(1,5);

poses_GT = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_GT.json');
disp('  pose GS loading')
poses_GS{1} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_GS_100.json');
poses_GS{2} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_GS_50.json');
poses_GS{3} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_GS_40.json');
poses_GS{4} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_GS_30.json');
poses_GS{5} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_GS_20.json');

disp('  pose Rerender loading')
poses_RR{1} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_RR_100.json');
poses_RR{2} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_RR_50.json');
poses_RR{3} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_RR_40.json');
poses_RR{4} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_RR_30.json');
poses_RR{5} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_RR_20.json');

disp('  pose MIXED loading')
poses_MX{1} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_MIXED_100.json');
poses_MX{2} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_MIXED_50.json');
poses_MX{3} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_MIXED_40.json');
poses_MX{4} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_MIXED_30.json');
poses_MX{5} = poseEstimAnal('W:\data\output\can_mtdew\result\refined_poses_MIXED_20.json');

%%
poseDiff_GS = cell(1,4);
poseDiff_RR = cell(1,5);
poseDiff_MX = cell(1,5);

disp('  pose GS evaluation')
for i = 1:4
    poseDiff_GS{i} = poseSetDiff(poses_GS{1}, poses_GS{i+1});
    summaryPoseDiff( poseDiff_GS{i}.dr, poseDiff_GS{i}.dt )
end

disp('  pose Rerender evaluation')
for i = 1:5
    poseDiff_RR{i} = poseSetDiff(poses_GS{1}, poses_RR{i});
    summaryPoseDiff( poseDiff_RR{i}.dr, poseDiff_RR{i}.dt )
end

disp('  pose Mixed evaluation')
for j = 1:5
    poseDiff_MX{j} = poseSetDiff(poses_GS{1}, poses_MX{j});
    summaryPoseDiff( poseDiff_MX{j}.dr, poseDiff_MX{j}.dt )
end