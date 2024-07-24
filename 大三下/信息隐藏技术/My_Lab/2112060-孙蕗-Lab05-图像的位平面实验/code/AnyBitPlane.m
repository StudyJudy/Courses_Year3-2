img = imread("Lena.bmp");
% 获取图像 img 的尺寸，
% 其中 m 表示图像的行数，n 表示图像的列数
[m,n]=size(img);
%创建一个大小与输入图像相同的空矩阵 c，用于存储位平面图像。
c=zeros(m,n);

for layer=1:8
    for i=1:m
        for j=1:n
            % 使用 bitget 函数从图像的每个像素中提取特定位（layer 指定的位），
            % 并将其存储在矩阵 c 中的相应位置。
            c(i,j)=bitget(img(i,j),layer);
        end
    end
figure
% 显示灰度图像 I，根据 I 中的像素值范围对显示进行转换。
% imshow 使用 [min(I(:)) max(I(:))] 作为显示范围。
% imshow 将 I 中的最小值显示为黑色，将最大值显示为白色
imshow(c,[]);
% pic=imshow(c,[]);
% frame = getframe(pic);
% im=framw.cdata;
title(['butterfly的第',num2str(layer),'个位平面']);
% imwrite(im,'Lena的第',num2str(layer),'个位平面');

end
