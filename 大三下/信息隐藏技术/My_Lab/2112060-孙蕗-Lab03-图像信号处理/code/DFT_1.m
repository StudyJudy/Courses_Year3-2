b=imread('butterfly.png');
figure;
I=im2bw(b);
imshow(I);
title('DFT二值图像');

figure;

%使用fft函数进行快速傅立叶变换
fa=fft2(I);
%fftshift函数调整fft函数的输出顺序，将零频位置移到频谱的中心
ffa=fftshift(fa);
image(abs(ffa));
title('DFT幅度谱');

figure;

%画网格曲面图
mesh(abs(ffa));
title('DFT幅度谱的能量分布');