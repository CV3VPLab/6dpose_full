function out_struct = py_dict_to_struct(py_dict)
    % 파이썬 딕셔너리를 매트랩 구조체(struct)로 변환하는 유틸리티 함수
    
    % 빈 구조체 생성
    out_struct = struct();
    
    
    % 파이썬 dict_keys를 매트랩 cell 배열로 변환
    keys_cell = cell(py.list(py_dict.keys()));
    
    for i = 1:length(keys_cell)
        % 키 추출 및 매트랩 문자열(char)로 변환
        key_str = char(keys_cell{i});
        
        % 값 추출
        val = py_dict.get(key_str);
        
        % --- 타입 변환 로직 ---
        % 파이썬 numpy 배열인 경우 매트랩 double 배열로 변환
        if isa(val, 'py.numpy.ndarray')
            val = double(val);
            
        % 파이썬 문자열인 경우 매트랩 char로 변환
        elseif isa(val, 'py.str')
            val = char(val);
            
        % 파이썬 int나 float인 경우 매트랩 double로 변환
        elseif isa(val, 'py.int') || isa(val, 'py.float')
            val = double(val);
            
        % 파이썬 리스트인 경우 매트랩 cell 배열로 변환 (기본적인 처리)
        elseif isa(val, 'py.list')
            val = cell(val);
        end
        
        % 구조체 필드에 값 할당
        out_struct.(key_str) = val;
    end
end