%离散余弦变换
[a,fs]=audioread('testaudio.wav');
da=dct(a);%离散余弦变换
a0=idct(da);%离散余弦逆变换4

figure('name','离散余弦变换');
subplot(3,1,1);plot(a);%原始波形
title('Raw wave')

subplot(3,1,2);plot(da);%dct()处理后
title('DCT')

subplot(3,1,3);plot(a0);%idct()重构后
title('Restored wave')