img = imread("butterfly.bmp");
% k = input("请输入n的值：");
% 获取图像 img 的尺寸，
% 其中 m 表示图像的行数，n 表示图像的列数
[m,n]=size(img);

% 创建了两个与原始图像尺寸相同的全零矩阵
% 分别用于存储低位和高位的位平面
low=zeros(m,n);
high=zeros(m,n);

for k=1:7
    % 迭代处理低位平面
    for ch=1:k
        for i=1:m
            for j=1:n
                % 将图像 img 的第 (i,j) 个像素的第 ch 位的值，通过 bitget 函数获取，
                % 并将其设置到 low 矩阵的对应像素的第 ch 位上。
                low(i,j)=bitset(low(i,j),ch,bitget(img(i,j),ch));
            end
        end
    end

    % 迭代处理高位平面
    for ch=k+1:8
        for i=1:m
            for j=1:n
                % 将图像 img 的第 (i,j) 个像素的第 ch 位的值，通过bitget函数获取，
                % 并将其设置到 high 矩阵的对应像素的第 ch 位上
                high(i,j)=bitset(high(i,j),ch,bitget(img(i,j),ch));
            end
        end
    end


figure;
subplot(1,2,1);
imshow(low,[]);
title(['butterfly的第1-',num2str(k),'个位平面']);

subplot(1,2,2);
imshow(high,[]);
title(['butterfly的第',num2str(k+1),'-8个位平面']);

end