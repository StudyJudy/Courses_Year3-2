a=imread('dog.jpg');
%转换为灰度图像
b=rgb2gray(a);

figure(1);
imshow(b);
title('灰度图像');
imwrite(b,'dog灰度图像.jpg');

I=im2bw(b);
figure(2);
imwrite(I,'dog二值图像.jpg')

%进行离散余弦变换
c=dct2(I);
imshow(c);
title('DCT变换系数');
imwrite(c,'dog_DCT变换系数.jpg');

figure;
%画网格曲面图
mesh(c);
title('DCT变换系数(立体图)');
imwrite(c,'dog_DCT变换系数（立体图）.jpg');
