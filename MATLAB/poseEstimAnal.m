function [T, loss] = poseEstimAnal(fn)

jsonText = fileread(fn);
jsonData = jsondecode(jsonText);
rp_tbl = struct2table(jsonData);
rp_tbl = sortrows(rp_tbl, "query");

T = [rp_tbl{:, "R"}, rp_tbl{:,"t"}];
loss = rp_tbl{:, "trackingLoss"};