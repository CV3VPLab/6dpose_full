%%
clear all
close all

fn = '..\data\can_data\3dgs_pepsi_pinset\pepsi_painted_canonical\point_cloud\iteration_30000\point_cloud.ply';

gaussians = read_3dgs_ply(fn);
figure, gsshow(gaussians)

