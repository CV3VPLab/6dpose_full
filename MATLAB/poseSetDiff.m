function res = poseSetDiff(poseRef, poseCmp)

res = struct("dr", cell2mat(poseRef.rvec) - cell2mat(poseCmp.rvec));
res.dt = cell2mat(poseRef.t) - cell2mat(poseCmp.t);