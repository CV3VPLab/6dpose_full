function rp_tbl = poseEstimAnal(fn)
%
% refine_poses.json

jsonText = fileread(fn);
jsonData = jsondecode(jsonText);
rp_tbl = struct2table(jsonData);
rp_tbl = sortrows(rp_tbl, "query");

rp_tbl.rvec = cellfun( @(x) {rotmat2vec3d(x)}, rp_tbl.R );
rp_tbl.rvec0 = cellfun( @(x) {rotmat2vec3d(x)}, rp_tbl.R0 );
rp_tbl.t = cellfun( @(x) {x'}, rp_tbl.t );
rp_tbl.t0 = cellfun( @(x) {x'}, rp_tbl.t0 );

dr = cell2mat(rp_tbl.rvec) - cell2mat(rp_tbl.rvec0);
dt = cell2mat(rp_tbl.t) - cell2mat(rp_tbl.t0);
rp_tbl.dr = mat2cell( dr, ones(1, length(dr)), [3] );
rp_tbl.dt = mat2cell( dt, ones(1, length(dt)), [3] );

summaryPoseDiff(dr, dt)


