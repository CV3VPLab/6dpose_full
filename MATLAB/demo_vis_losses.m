%%
clear all
close all

pyenv(Version="C:\Users\admin\anaconda3\envs\pyenv\python.exe");

%%
path_res = 'W:\data\output\can_fanta\result';

npy_file_info = dir( fullfile(path_res, '*.npy') );
nImgs = size(npy_file_info, 1);

for ii = 1:nImgs
    filepath = fullfile( path_res, npy_file_info(ii).name );
    l = py.numpy.load( filepath, pyargs('allow_pickle', true) );
    
    losses = [];
    
    for i = 0:double(l.size)-1
        losses = [losses, py_dict_to_struct(l.item(int32(i)))];
    end
    
    plot_losses(struct2table(losses))
    title(npy_file_info(ii).name)
end

