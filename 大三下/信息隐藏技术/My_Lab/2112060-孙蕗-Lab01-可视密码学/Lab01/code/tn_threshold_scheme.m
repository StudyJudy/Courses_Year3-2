function tn_threshold_scheme
    clear;
    close all;

    width = 256;
    height = 256;
    t = 4;
    n = 5;
    % 选择用于恢复原图的t张阴影图
    chosen_list = [2,3,4,5];
    % 模数
    mod_value = 257;

    sub_width = width / t ^ (0.5);
    sub_height = height / t ^ (0.5);
    
    path = input("输入秘密图像:",'s');
    % path =
    % E:/Download/MATLAB/information_hide/Lab01_Visual_Cryptography/lena.bin
    original_img = fopen(path,'rb');
    linear_img = fread(original_img,'uint8');
    original_img = uint8(reshape(linear_img, height, width)');

    figure;
    imshow(original_img);
    if exist('figure', 'dir') ~= 7
    % 如果不存在，则创建文件夹
        mkdir('figure');
    end
    imwrite(original_img,'./figure/original_img.bmp');

    % 将秘密图像均匀分割成t幅子图像
    secret_img = reshape(linear_img, [], t);
    % 由t幅子图像生成n幅阴影图像
    shadow_img = gen_shadow_img(secret_img, n, mod_value, sub_width, sub_height);
    % 
    merge_shadow_img(shadow_img,chosen_list,height,width,mod_value);

end

% gen_shadow_img - 通过t幅子图像生成n幅阴影图像
%
% 输入:
%   - secret_img: 待分解的秘密图像
%   - n: 阴影图像数
%   - mod_value: 模数
%   - sub_wdith/height: 阴影图像宽/高
% 输出:
%   - shadow_img: 生成的阴影图像
function shadow_img = gen_shadow_img(secret_img, n, mod_value, sub_width, sub_height)

    Size = int32(size(secret_img));
    secret_img = int32(secret_img);
    n = int32(n);
    mod_value = int32(mod_value);

    shadow_img = int32(zeros(Size(1),n));

    % 生成阴影图像
    for i = 1 : n
        for j = 1 : Size(2)
            shadow_img(:, i) = shadow_img(:, i) + mod(secret_img(:, j) * i ^ (j - 1), mod_value);
        end
    end

    shadow_img = mod(shadow_img, mod_value);
    
    Size = size(shadow_img);
    % 显示并保存阴影图像
    for i = 1 : Size(2)
        figure;
        imshow(uint8(reshape(shadow_img(:, i),sub_width,sub_height)));
        title(['shadow\_img' num2str(i)]);
        imwrite(uint8(reshape(shadow_img(:, i),sub_width,sub_height)), ['./figure/shadow_img_' num2str(i) '.bmp']);
    end
end

% merge_shadow_img - 将t张阴影图合成为秘密图像：
%
% 输入:
%   - shadow_img: 阴影图像集合
%   - chosen_list: 被选择的t张阴影图
%   - width/height: 秘密图像的宽/高
%   - mod_value: 模数
% 输出:
%   - merged_img: 合成后的原始秘密图像
function merged_img = merge_shadow_img(shadow_img, chosen_list, height, width, mod_value)
    
    % 选出用于合成的t张阴影图
    for i = 1 : length(chosen_list)
        chosen_shadow_img(:, i) = shadow_img(:, chosen_list(i));
    end

    shadow_img_Size = size(chosen_shadow_img);
    m = ones(shadow_img_Size(2),shadow_img_Size(2));
    for i = 1 : shadow_img_Size(2)
        m(:, i) = m(:, i) .* (chosen_list.^(i-1))';
    end
    
    % 进度条
    h = waitbar(0,'Merging image...');
    for i = 1 : shadow_img_Size(1)
        waitbar(i/shadow_img_Size(1),h,sprintf('Merging image... %d%%', round(i/shadow_img_Size(1)*100)));
        % 求解线性方程组
        merged_img(i,:) = get_mod(inv(sym(m)) * chosen_shadow_img(i, :)', mod_value)';
    end
    close(h);

    merged_img = reshape(merged_img(:),height,width)';

    figure;
    imshow(uint8(merged_img));
    title('merged\_img');
    imwrite(uint8(merged_img), './figure/merged_img.bmp');
end

% get_mod - 计算
% 输入:
%   - base: 基数
%   - mod_value: 模数
% 输出:
%   - result: 
function result = get_mod(base, mod_value)
    [N,D] = numden(base);
    N = double(N);
    D = double(D);
    result = mod(N.*power_mod(D, -1, mod_value),mod_value);
end

% power_mod - 模幂运算。
%
% 输入:
%   - base: 基数。
%   - exp: 指数。
%   - mod_value: 模数。
%
% 输出:
%   - result: 幂模运算的结果。
function result = power_mod(base, exp, mod_value)
    [rows, cols] = size(base);
    base = mod(base, mod_value);
    
    if (exp < 0)
        exp = -exp;
        for row = 1 : rows
            for col = 1 : cols
                base(row, col) = mod_inverse(base(row, col), mod_value);
            end
        end 
    end

    result = ones(rows, cols);
    for row = 1 : rows
        for col = 1 : cols
            base_element = base(row, col);
            exp_element = exp;
            while (exp_element > 0)
                if mod(exp_element, 2) == 1
                    result(row, col) = mod(result(row, col) * base_element, mod_value);
                end
                base_element = mod(base_element^2, mod_value);
                exp_element = floor(exp_element / 2);
            end
        end
    end
end

% mod_inverse - 计算模的乘法逆元素
function inverse = mod_inverse(num, mod_value)
    % 使用扩展欧几里得算法计算乘法逆元素
    [~, inverse, ~] = gcd(num, mod_value);
    inverse = mod(inverse, mod_value);
end






