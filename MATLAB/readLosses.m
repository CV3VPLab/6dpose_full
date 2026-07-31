function losses = readLosses(pyfile)

l = py.numpy.load( pyfile, pyargs('allow_pickle', true));
itrs = int32(l.size);
assert( itrs > 0 );

losses = {struct2table(py_dict_to_struct(l.item(int32(0))))};
nfields = size(losses{1}, 2);
ncats = 1;

for i = 1:itrs-1
    lt = struct2table(py_dict_to_struct(l.item(i)));
    idx = find( nfields == size(lt, 2) );
    if isempty(idx)
        ncats = ncats + 1;
        losses{ncats} = lt;
        nfields(ncats) = size(lt, 2);
    else
        assert(isscalar(idx));
        losses{idx} = [losses{idx}; lt]; 
    end
end    

