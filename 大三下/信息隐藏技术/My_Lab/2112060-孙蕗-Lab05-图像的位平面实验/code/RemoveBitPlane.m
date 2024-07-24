img = imread("butterfly.bmp");

% 获取图像 img 的尺寸，
% 其中 m 表示图像的行数，n 表示图像的列数
[m,n]=size(img);

for layer=1:7
    % 遍历前layer个位平面图像的每个像素
    for ch=1:layer
        for i=1:m
            for j=1:n
                % 将图像 img 的第 (i,j) 个像素的第 ch 位设置为 0
                img(i,j)=bitset(img(i,j),ch,0);
            end
        end
    end
    
figure;
imshow(img,[]);
title(['butterfly去除前',num2str(layer),'个位平面']);

end