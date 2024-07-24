%快速离散傅里叶变换
[x,fs]=audioread('testaudio.wav');
fx=fft(x);%fft函数 快速离散傅里叶变换

x0=ifft(fx);%逆变换

figure('name','快速傅里叶变换FFT');

subplot(3,1,1);
plot(x);
title('Raw Audio')

subplot(3,1,2);
plot(abs(fftshift(fx)));
xlabel('Time');
ylabel('Amplitude');
title('The Wave form of signal FFT');

grid on;

subplot(3,1,3);
plot(x0);
title('Restored audio')