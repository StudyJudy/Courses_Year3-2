% 利用LSB方法实现在wav文件中的信息隐藏与提取
clc;  
clear; 
close all;

% 打开原始音频文件并读取数据
fid = fopen('male_voice.wav', 'rb');
origin_audio = fread(fid, inf, 'uint8');
fclose(fid);

% 计算可隐藏信息的长度（去除文件头44字节）
len = length(origin_audio) - 44;

% 秘密图像
ScrietImg = imread("Pig.jpg");
GrayScrietImg = im2gray(ScrietImg);
BinScrietImg = imbinarize(GrayScrietImg);

% 读取要隐藏的信息（二值图像）
[rows, cols] = size(BinScrietImg);
hidden_bits = BinScrietImg(:);
if rows * cols > len
    error('音频载体太小，请更换载体');
end

% 将隐藏信息嵌入到音频数据的44字节后面
modified_audio = origin_audio;
for k = 1 : rows * cols
    modified_audio(44 + k) = bitset(modified_audio(44 + k), 1, hidden_bits(k));
end

% 将带有隐藏信息的音频数据写入到新文件
fid = fopen('modified_audio.wav', 'wb');
fwrite(fid, modified_audio, 'uint8');
fclose(fid);

% 读取原始音频和带有隐藏信息的音频
[raw_audio, fs_raw] = audioread('male_voice.wav');
[modified_audio, fs_mod] = audioread('modified_audio.wav');

% 计算时间向量
t_raw = (0:length(raw_audio)-1) / fs_raw;
t_mod = (0:length(modified_audio)-1) / fs_mod;

% 绘制原始音频波形图
figure;
subplot(2,1,1);
plot(t_raw, raw_audio);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Raw Audio Waveform');

% 绘制带有隐藏信息的音频波形图
subplot(2,1,2);
plot(t_mod, modified_audio);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Modified Audio Waveform');

% 调整子图布局
sgtitle('Comparison of Raw and Modified Audio Waveforms');

% 提取隐藏信息
% 打开带有隐藏信息的音频文件并读取数据
fid = fopen('modified_audio.wav', 'rb');
audio_data = fread(fid, inf, 'uint8');
fclose(fid);

% 提取隐藏信息的二值图像
hidden_bits = zeros(rows * cols, 1);
for k = 1 : rows * cols
    hidden_bits(k) = bitget(audio_data(44 + k), 1);
end

% 将一维隐藏信息转换为二维图像
hidden_image = reshape(hidden_bits, rows, cols);

% 显示提取的隐藏信息图像
figure;
imshow(hidden_image,[]);
title('Extracted Hidden Image');
imwrite(hidden_image,"hidden_image.bmp");


