%%
clear all
close all

pyenv(Version="C:\Users\choik\anaconda3\envs\gsplat\python.exe");

path_res = 'S:\data\output\can_mtdew\result';

%%
imgnum = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14];

for ii = 1:length(imgnum)
    imgname = sprintf('%05d', imgnum(ii));
    l = py.numpy.load( fullfile(path_res, [imgname, '_L.npy']), pyargs('allow_pickle', true));
    
    losses = [];
    
    for i = 0:double(l.size)-1
        losses = [losses, py_dict_to_struct(l.item(int32(i)))];
    end
    
    plot_losses(losses)
    title(imgname)
end

