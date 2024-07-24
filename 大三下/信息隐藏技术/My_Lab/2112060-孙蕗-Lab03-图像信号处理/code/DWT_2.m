%图像的二级小波变换
b=imread("Lena.jpg");
a=im2bw(b);%二值图
nbcol=size(a,1);%获取二值图像的行数

[ca1,ch1,cv1,cd1]=dwt2(a,'db4');
[ca2,ch2,cv2,cd2]=dwt2(ca1,'db4');
cod_ca1=wcodemat(ca1,nbcol);
cod_ch1=wcodemat(ch1,nbcol);
cod_cv1=wcodemat(cv1,nbcol);
cod_cd1=wcodemat(cd1,nbcol);

cod_ca2=wcodemat(ca2,nbcol);
cod_ch2=wcodemat(ch2,nbcol);
cod_cv2=wcodemat(cv2,nbcol);
cod_cd2=wcodemat(cd2,nbcol);

tt=[cod_ca2,cod_ch2;cod_cv2,cod_cd2];
tt=imresize(tt,[size(ca1,1),size(ca1,1)]);

figure;
image([tt,cod_ch1;cod_cv1,cod_cd1]);
title('Lena图像二级小波分解');