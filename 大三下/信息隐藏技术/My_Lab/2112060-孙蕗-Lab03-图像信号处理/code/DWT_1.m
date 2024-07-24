%图像的一级小波变换
b=imread('butterfly.png');
a=rgb2gray(b);
nbcol=size(a,1);%获取灰度图像的行数

[ca1,ch1,cv1,cd1]=dwt2(a,'db4');
cod_ca1=wcodemat(ca1,nbcol);
cod_ch1=wcodemat(ch1,nbcol);
cod_cv1=wcodemat(cv1,nbcol);
cod_cd1=wcodemat(cd1,nbcol);

image([cod_ca1,cod_ch1;cod_cv1,cod_cd1]);
imwrite([cod_ca1,cod_ch1;cod_cv1,cod_cd1],'butterfly图像一级小波分解.jpg')

% image(cod_ca1);

figure('name','butterfly_cod_ca1_近似部分');
image(cod_ca1);
% imwrite(cod_ca1,'butterfly_cod_ca1_近似部分.jpg');

figure('name','butterfly_cod_ch1_水平方向细节部分');
image(cod_ch1);
% imwrite(cod_ch1,'butterfly_cod_ch1_水平方向细节部分.jpg');

figure('name','butterfly_cod_cv1_竖直方向细节部分');
image(cod_cv1);
% imwrite(cod_cv1,'butterfly_cod_ca1_竖直方向细节部分.jpg');

figure('name','butterfly_cod_cd1_对角线方向细节部分');
image(cod_cd1);
% imwrite(cod_cd1,'butterfly_cod_cd1_对角线方向细节部分.jpg');



