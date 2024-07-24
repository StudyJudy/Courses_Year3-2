%db4一级小波分解与重构DWT
[a,fs]=audioread('testaudio.wav');
[ca1,cd1]=dwt(a(:,1),'db4');%db4一级小波分解，取原音频的
a0=idwt(ca1,cd1,'db4',length(a(:,1)));%db4一级小波分解重构

figure('name','一级小波分解与重构DWT');
ax(1)=subplot(4,1,1);
plot(a(:,1));%原始波形
title('Raw wave')
grid on;

ax(2)=subplot(4,1,2);
plot(cd1);%细节分量
title('Component of Detail')
grid on;

ax(3)=subplot(4,1,3);
plot(ca1);%近似分量
title('Approximate Component')
grid on;

ax(4)=subplot(4,1,4);
plot(a0);%一级分解重构的结果
title('Recovered wave')
grid on;

linkaxes(ax,'x');