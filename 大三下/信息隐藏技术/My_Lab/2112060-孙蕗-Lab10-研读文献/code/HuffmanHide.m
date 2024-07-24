%% 主函数：
clear
clc
I = imread('Baboon.tif');

origin_I = double(I);  %double(I)是将读入的图像I的uint8数据转换为double类型的数据
% 产生二进制秘密数据
num = 10000;
rand('seed',0); %设置种子
D = round(rand(1,num)*1); %产生稳定随机数
% 使用图片代替随机数据
% J = imread('Baboon.tif');
% D = dec2bin(J);%将待嵌入数据转换成二进制形式
% D = strcat(char(D)', '');%将其转换成字符数组
% D = str2num(D(:));%将其转换成整数数组
% num=numel(D);%统计待嵌入数据总长度

% 设置图像加密密钥及数据加密密钥
Image_key = 1; 
Data_key = 2;
% 设置参数(方便实验修改)
ref_x = 1; %用来作为参考像素的行数
ref_y = 1; %用来作为参考像素的列数
% 图像加密及数据嵌入
[encrypt_I,stego_I,emD] = Encrypt_Embed(origin_I,D,Image_key,Data_key,ref_x,ref_y);
% 数据提取及图像恢复
num_emD = length(emD);
if num_emD > 0  %表示有空间嵌入数据
    %--------在加密标记图像中提取信息--------%
    [Side_Information,Refer_Value,Encrypt_exD,Map_I,sign] = Extract_Data(stego_I,num,ref_x,ref_y);
    if sign == 1 %表示能完全提取辅助信息
        %---------------数据解密----------------%
        [exD] = Encrypt_Data(Encrypt_exD,Data_key);
        %---------------图像恢复----------------%
        [recover_I] = Recover_Image(stego_I,Image_key,Side_Information,Refer_Value,Map_I,num,ref_x,ref_y);
        %---------------图像对比----------------%
        figure;
        subplot(221);imshow(origin_I,[]);title('原始图像');
        subplot(222);imshow(encrypt_I,[]);title('加密图像');
        subplot(223);imshow(stego_I,[]);title('载密图像');
        subplot(224);imshow(recover_I,[]);title('恢复图像');
        %---------------结果记录----------------%
        [m,n] = size(origin_I);
        bpp = num_emD/(m*n);
        %---------------结果判断----------------%
        check1 = isequal(emD,exD);
        check2 = isequal(origin_I,recover_I);
        if check1 == 1
            disp('提取数据与嵌入数据完全相同！')
        else
            disp('Warning！数据提取错误！')
        end
        if check2 == 1
            disp('重构图像与原始图像完全相同！')
        else
            disp('Warning！图像重构错误！')
        end
        %---------------结果输出----------------%
        if check1 == 1 && check2 == 1
            disp(['嵌入容量为 : ' num2str(num_emD)])
            disp(['嵌入率为 : ' num2str(bpp)])
            fprintf(['该测试图像------------ OK','\n\n']);
        else
            fprintf(['该测试图像------------ ERROR','\n\n']);
        end     
    else
        disp('无法提取全部辅助信息！')
        fprintf(['该测试图像------------ ERROR','\n\n']);
    end
else
    disp('辅助信息大于总嵌入量，导致无法存储数据！') 
    fprintf(['该测试图像------------ ERROR','\n\n']);
end

%% =====================辅助函数======================
% 二进制转十进制：
function [value] = Binary_Decimalism(bin2_8)
    % 函数说明：将二进制数组转换成十进制整数
    % 输出：bin2_8（二进制数组）
    % 输入：value（十进制整数）
    value = 0;
    len = length(bin2_8);
    for i=1:len
        value = value + bin2_8(i)*(2^(len-i));
    end
end
% 十进制转二进制：
function [bin2_8] = Decimalism_Binary(value)
    % 函数说明：将十进制灰度像素值转换成8位二进制数组
    % 输入：value（十进制灰度像素值）
    % 输出：bin2_8（8位二进制数组）
    bin2_8 = dec2bin(value)-'0';
    if length(bin2_8) < 8
        len = length(bin2_8);
        B = bin2_8;
        bin2_8 = zeros(1,8);
        for i=1:len
            bin2_8(8-len+i) = B(i); %不足8位前面补充0
        end 
    end
end

%% ======================图像预测阶段======================

function [origin_PV_I] = Predictor_Value(origin_I,ref_x,ref_y)  
    % 函数说明：计算origin_I的预测值
    % 输入：origin_I（原始图像）,ref_x,ref_y（参考像素的行列数）
    % 输出：origin_PV_I（原始图像的预测值）
    [row,col] = size(origin_I); %计算origin_I的行列值
    origin_PV_I = origin_I;  %构建存储origin_I预测值的容器
    for i=ref_x+1:row  %前ref_x行作为参考像素，不用预测
        for j=ref_y+1:col  %前ref_y列作为参考像素，不用预测
            a = origin_I(i-1,j);
            b = origin_I(i-1,j-1);
            c = origin_I(i,j-1);
            if b <= min(a,c)
                origin_PV_I(i,j) = max(a,c);
            elseif b >= max(a,c)
                origin_PV_I(i,j) = min(a,c);
            else
                origin_PV_I(i,j) = a + c - b;
            end
        end
    end
end
% ======================图像预测阶段======================



%% =====================自适应的哈夫曼编码======================
% 哈夫曼编码：
function [Code,Code_Bin] = Huffman_Code(num_Map_origin_I)
    % 函数说明：用变长编码(多位0/1编码)表示像素值的标记类别
    % 输入：num_Map_origin_I（像素值标记类别的统计情况）
    % 输出：Code（映射关系）,Code_Bin（Code的二进制表示）
    % 备注：用{00,01,100,101,1100,1101,1110,11110,11111}这9中编码来表示9种标记类别
    % 规则：标记类别中像素的个数越多，则用来表示其类别的编码长度越短
    % {00,01,100,101,1100,1101,1110,11110,11111}→{0,1,4,5,12,13,14,30,31}
    % 求其映射编码关系
    %Code = [-1,0;-1,1;-1,4;-1,5;-1,12;-1,13;-1,14;-1,30;-1,31]; 
    %此处的-1仅有一个标记作用，也可以修改为其他第二列未出现的值，最终都会被替换成映射关系
    Code = [-2,0;-2,1;-2,4;-2,5;-2,12;-2,13;-2,14;-2,30;-2,31]; 
    for i=1:9
        drder=1;
        for j=1:9
            if num_Map_origin_I(i,2) < num_Map_origin_I(j,2)
                drder = drder + 1; %排序寻找最小值
            end
        end
        while Code(drder) ~= -2 %防止两种标记类别中像素的个数相等
            drder = drder + 1; 
        end
        Code(drder,1) = num_Map_origin_I(i,1); %第一列从小到大排列了出现的标记顺序
    end
    % 将Map映射关系用二进制比特流表示
    Code_Bin = zeros();
    t = 0; %计数
    for i=0:8
        for j=1:9
            if Code(j,1) == i
                value = Code(j,2);
            end
        end
        if value == 0
            Code_Bin(t+1) = 0;
            Code_Bin(t+2) = 0;
            t = t+2;
        elseif value == 1
            Code_Bin(t+1) = 0;
            Code_Bin(t+2) = 1;
            t = t+2;
        else 
            add = ceil(log2(value+1)); %表示标记编码的长度
            Code_Bin(t+1:t+add) = dec2bin(value)-'0'; %将value转换成二进制数组
            t = t + add;
        end     
    end
end

% 哈夫曼解码：
function [value,this_end] = Huffman_DeCode(Binary,last_end)
    % 求二进制比特流Binary中下一个Huffman编码转换成的整数值
    % 输入：Binary（二进制映射序列）,last_end（上一个映射结束的位置）
    % 输出：value（十进制整数值）→{0,1,4,5,12,13,14,30,31},end（本次结束的位置）
    len = length(Binary);
    i = last_end+1;
    t = 0; %计数
    if i <= len
        if i+1<=len && Binary(i)==0 %→0
            t = t + 1;
            if Binary(i+1) == 0 %→00表示0
                t = t + 1;
                value = 0;
            elseif Binary(i+1) == 1 %→01表示1
                t = t + 1;
                value = 1;
            end
        else  % Binary(i)==1
            if i+2<=len && Binary(i+1)==0 %→10
                t = t + 2;
                if Binary(i+2) == 0  %→100表示4
                    t = t + 1;
                    value = 4;
                elseif Binary(i+2) == 1 %→101表示5
                    t = t + 1;
                    value = 5;
                end
            else % Binary(i+1)==1
                if i+3<=len && Binary(i+2)==0  %→110
                    t = t + 3;
                    if Binary(i+3) == 0  %→1100表示12
                        t = t + 1;
                        value = 12;
                    elseif Binary(i+3) == 1  %→1101表示13
                        t = t + 1;
                        value = 13;
                    end
                else % Binary(i+2)==1
                    if i+3 <= len
                        t = t + 3;
                        if Binary(i+3) == 0  %→1110表示14
                            t = t + 1;
                            value = 14;
                        elseif i+4<=len && Binary(i+3)==1  %→1111
                            t = t + 1;
                            if Binary(i+4) == 0  %→11110表示30
                                t = t + 1;
                                value = 30;
                            elseif Binary(i+4) == 1  %→11111表示31
                                t = t + 1;
                                value = 31;
                            end
                        else
                            t = 0;   
                            value = -1; %表示辅助信息长度不够，无法恢复下一个Huffman编码
                        end
                    else
                        t = 0;
                        value = -1; %表示辅助信息长度不够，无法恢复下一个Huffman编码
                    end
                end
            end
        end
    else
        t = 0;               
        value = -1; %表示辅助信息长度不够，无法恢复下一个Huffman编码
    end
    this_end = last_end + t;
end
% =====================自适应的哈夫曼编码======================


%% ======================图像加密阶段======================
% 图像异或加密：
function [encrypt_I] = Encrypt_Image(origin_I,Image_key)
    % 函数说明：对图像origin_I进行bit级异或加密
    % 输入：origin_I（原始图像）,Image_key（图像加密密钥）
    % 输出：encrypt_I（加密图像）
    [row,col] = size(origin_I); %计算origin_I的行列值
    encrypt_I = origin_I;  %构建存储加密图像的容器
    % 根据密钥生成与origin_I大小相同的随机矩阵
    rand('seed',Image_key); %设置种子
    E = round(rand(row,col)*255); %随机生成row*col矩阵
    % 根据E对图像origin_I进行bit级加密
    for i=1:row
        for j=1:col
            encrypt_I(i,j) = bitxor(origin_I(i,j),E(i,j));
        end
    end
end

% 信息异或加密：
function [Encrypt_D] = Encrypt_Data(D,Data_key)
    % 函数说明：对原始秘密信息D进行bit级异或加密
    % 输入：D（原始秘密信息）,Data_key（数据加密密钥）
    % 输出：Encrypt_D（加密的秘密信息）
    num_D = length(D); %求嵌入数据D的长度
    Encrypt_D = D;  %构建存储加密秘密信息的容器
    % 根据密钥生成与D长度相同的随机0/1序列
    rand('seed',Data_key); %设置种子
    E = round(rand(1,num_D)*1); %随机生成长度为num_D的0/1序列
    % 根据E对原始秘密信息D进行异或加密
    for i=1:num_D  
        Encrypt_D(i) = bitxor(D(i),E(i));
    end
end

% ======================图像加密阶段======================

%% ======================位图嵌入阶段======================

% 位图标记：
function [Map_origin_I] = Category_Mark(origin_PV_I,origin_I,ref_x,ref_y)
    % 函数说明：对每个像素值进行标记，即生成原始图像的位图
    % 输入：origin_PV_I（预测值），origin_I（原始值）,ref_x,ref_y（参考像素的行列数）
    % 输出：Map_origin_I（像素值的标记分类,即位置图）
    [row,col] = size(origin_I); %计算origin_I的行列值
    Map_origin_I = origin_I;  %构建存储origin_I标记的容器
    for i=1:row
        for j=1:col
            if i<=ref_x || j<=ref_y %前ref_x行、前ref_y列作为参考像素，不标记
                Map_origin_I(i,j) = -1;   %标记为-1是为了与下面产生的t=7时ca=0情况分开，两种情况都不能嵌入数据，但是参考像素不必恢复，非参考像素需要在恢复操作中被遍历
            else
                x = origin_I(i,j); %原始值
                pv = origin_PV_I(i,j); %预测值
                for t=7:-1:0  
                    if floor(x/(2^t)) ~= floor(pv/(2^t))  % floor(x) 函数向下取整
                        ca = 8-t-1; %用来记录像素值的标记类别  -1是因为不同的那一位也可以嵌入数据
                        break;
                    else
                        ca = 8; 
                    end
                end
                Map_origin_I(i,j) = ca; %表示有ca位MSB相同，即可以嵌入ca位信息
            end        
        end
    end

end

% 位图转化为二进制数组：
function [Map_Bin] = Map_Binary(Map_origin_I,Code)
    % 函数说明：将位图Map_origin_I转换成二进制数组Map
    % 输入：Map_origin_I（原始图像的位置图）,Code（映射关系）
    % 输出：Map_Bin（原始图像位置图的二进制数组）
    [row,col] = size(Map_origin_I); %计算Map_origin_II的行列值
    Map_Bin = zeros();
    t = 0; %计数，二进制数组的长度
    for i=1:row 
        for j=1:col
            if Map_origin_I(i,j) == -1 %标为-1的点是作为参考像素，不统计
                continue;
            end
            for k=1:9
                if Map_origin_I(i,j) == Code(k,1)
                    value = Code(k,2);
                    break;
                end
            end
            if value == 0
                Map_Bin(t+1) = 0;
                Map_Bin(t+2) = 0;
                t = t+2;
            elseif value == 1
                Map_Bin(t+1) = 0;
                Map_Bin(t+2) = 1;
                t = t+2;
            else
                add = ceil(log2(value+1)); %表示标记编码的长度
                Map_Bin(t+1:t+add) = dec2bin(value)-'0'; %将value转换成二进制数组
                t = t + add;
            end 
        end
    end
end
% ======================位图嵌入阶段======================

%% ======================信息嵌入阶段======================

% 根据位图嵌入秘密信息和辅助信息：
function [stego_I,emD] = Embed_Data(encrypt_I,Map_origin_I,Side_Information,D,Data_key,ref_x,ref_y)
    % 函数说明：根据位置图将辅助信息和秘密信息嵌入到加密图像中
    % 输入：encrypt_I（加密图像）,Map_origin_I（位置图）,Side_Information（辅助信息）,D（秘密信息）,Data_key（数据加密密钥）,ref_x,ref_y（参考像素的行列数）
    % 输出：stego_I（加密标记图像）,emD（嵌入的数据）
    stego_I = encrypt_I;
    [row,col] = size(encrypt_I); %统计encrypt_I的行列数
    % 对原始秘密信息D进行加密
    [Encrypt_D] = Encrypt_Data(D,Data_key);
    % 将前ref_y列、前ref_x行的参考像素记录下来，放在秘密信息之前嵌入图像中
    Refer_Value = zeros(); %记录参考像素的数组
    t = 0; %计数
    for i=1:row
        for j=1:ref_y  %前ref_y列
            value = encrypt_I(i,j);
            [bin2_8] = Decimalism_Binary(value); %将十进制整数转换成8位二进制数组
            Refer_Value(t+1:t+8) = bin2_8; %因为t=0，所以从t+1开始
            t = t + 8; 
        end
    end
    for i=1:ref_x  %前ref_x行
        for j=ref_y+1:col
            value = encrypt_I(i,j);
            [bin2_8] = Decimalism_Binary(value); %将十进制整数转换成8位二进制数组
            Refer_Value(t+1:t+8) = bin2_8;
            t = t + 8; 
        end
    end 
    % 辅助量
    num_D = length(D); %求秘密信息D的长度
    num_emD = 0; %计数，统计嵌入秘密信息的个数
    num_S = length(Side_Information); %求辅助信息Side_Information的长度
    num_side = 0;%计数，统计嵌入辅助信息的个数
    num_RV = length(Refer_Value); %参考像素二进制序列信息的长度
    num_re = 0; %计数，统计嵌入参考像素二进制序列信息的长度
    % 先在前ref_y列、前ref_x行的参考像素中存储辅助信息
    for i=1:row
        for j=1:ref_y  %前ref_y列
            bin2_8 = Side_Information(num_side+1:num_side+8);
            [value] = Binary_Decimalism(bin2_8); %将8位二进制数组转换成十进制整数
            stego_I(i,j) = value;
            num_side = num_side + 8;
        end
    end
    for i=1:ref_x  %前ref_x行
        for j=ref_y+1:col
            bin2_8 = Side_Information(num_side+1:num_side+8);
            [value] = Binary_Decimalism(bin2_8); %将8位二进制数组转换成十进制整数
            stego_I(i,j) = value;
            num_side = num_side + 8;
        end
    end
    % 再在其余位置嵌入辅助信息、参考像素和秘密数据
    for i=ref_x+1:row  
        for j=ref_y+1:col 
            if num_emD >= num_D %秘密数据已嵌完
                break;
            end
            %------表示这个像素点可以嵌入 1 bit信息------%
            if Map_origin_I(i,j) == 0  %Map=0表示原始像素值的第1MSB与其预测值相反
                if num_side < num_S %辅助信息没有嵌完
                    num_side = num_side + 1;
                    stego_I(i,j) = mod(stego_I(i,j),2^7) + Side_Information(num_side)*(2^7); %替换1位MSB
                else
                    if num_re < num_RV %参考像素二进制序列信息没有嵌完
                        num_re = num_re + 1;
                        stego_I(i,j) = mod(stego_I(i,j),2^7) + Refer_Value(num_re)*(2^7); %替换1位MSB
                    else %最后嵌入秘密信息
                        num_emD = num_emD + 1;
                        stego_I(i,j) = mod(stego_I(i,j),2^7) + Encrypt_D(num_emD)*(2^7); %替换1位MSB
                    end       
                end
            %------表示这个像素点可以嵌入 2 bit信息------%
            elseif Map_origin_I(i,j) == 1  %Map=1表示原始像素值的第2MSB与其预测值相反  
                if num_side < num_S %辅助信息没有嵌完
                    if num_side+2 <= num_S %2位MSB都用来嵌入辅助信息
                        num_side = num_side + 2;
                        stego_I(i,j) = mod(stego_I(i,j),2^6) + Side_Information(num_side-1)*(2^7) + Side_Information(num_side)*(2^6); %替换2位MSB
                    else
                        num_side = num_side + 1; %1bit辅助信息
                        num_re = num_re + 1; %1bit参考像素二进制序列信息
                        stego_I(i,j) = mod(stego_I(i,j),2^6) + Side_Information(num_side)*(2^7) + Refer_Value(num_re)*(2^6); %替换2位MSB
                    end
                else
                    if num_re < num_RV %参考像素二进制序列信息没有嵌完
                        if num_re+2 <= num_RV %2位MSB都用来嵌入参考像素二进制序列信息   
                            num_re = num_re + 2;
                            stego_I(i,j) = mod(stego_I(i,j),2^6) + Refer_Value(num_re-1)*(2^7) + Refer_Value(num_re)*(2^6); %替换2位MSB
                        else
                            num_re = num_re + 1; %1bit参考像素二进制序列信息
                            num_emD = num_emD + 1; %1bit秘密信息
                            stego_I(i,j) = mod(stego_I(i,j),2^6) + Refer_Value(num_re)*(2^7) + Encrypt_D(num_emD)*(2^6); %替换2位MSB
                        end
                    else
                        if num_emD+2 <= num_D
                            num_emD = num_emD + 2; %2bit秘密信息
                            stego_I(i,j) = mod(stego_I(i,j),2^6) + Encrypt_D(num_emD-1)*(2^7) + Encrypt_D(num_emD)*(2^6); %替换2位MSB
                        else
                            num_emD = num_emD + 1; %1bit秘密信息
                            stego_I(i,j) = mod(stego_I(i,j),2^7) + Encrypt_D(num_emD)*(2^7); %替换1位MSB
                        end   
                    end
                end
            %------表示这个像素点可以嵌入 3 bit信息------%
            elseif Map_origin_I(i,j) == 2  %Map=2表示原始像素值的第3MSB与其预测值相反
                bin2_8 = zeros(1,8); %用来记录要嵌入的信息，少于8位的低位(LSB)默认为0
                if num_side < num_S %辅助信息没有嵌完
                    if num_side+3 <= num_S %3位MSB都用来嵌入辅助信息
                        bin2_8(1:3) = Side_Information(num_side+1:num_side+3); 
                        num_side = num_side + 3;
                        [value] = Binary_Decimalism(bin2_8);
                        stego_I(i,j) = mod(stego_I(i,j),2^5) + value; %替换3位MSB
                    else
                        t = num_S - num_side; %剩余辅助信息个数
                        bin2_8(1:t) = Side_Information(num_side+1:num_S); %tbit辅助信息
                        num_side = num_side + t;
                        bin2_8(t+1:3) = Refer_Value(num_re+1:num_re+3-t); %(3-t)bit参考像素二进制序列信息
                        num_re = num_re + 3-t;
                        [value] = Binary_Decimalism(bin2_8);
                        stego_I(i,j) = mod(stego_I(i,j),2^5) + value; %替换3位MSB
                    end
                else
                    if num_re < num_RV  %参考像素二进制序列信息没有嵌完
                        if num_re+3 <= num_RV %3位MSB都用来嵌入参考像素二进制序列信息
                            bin2_8(1:3) = Refer_Value(num_re+1:num_re+3); 
                            num_re = num_re + 3;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^5) + value; %替换3位MSB
                        else
                            t = num_RV - num_re; %剩余参考像素二进制序列信息个数
                            bin2_8(1:t) = Refer_Value(num_re+1:num_RV); %tbit参考像素二进制序列信息
                            num_re = num_re + t;
                            bin2_8(t+1:3) = Encrypt_D(num_emD+1:num_emD+3-t); %(3-t)bit秘密信息
                            num_emD = num_emD + 3-t;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^5) + value; %替换3位MSB
                        end 
                    else
                        if num_emD+3 <= num_D
                            bin2_8(1:3) = Encrypt_D(num_emD+1:num_emD+3); %3bit秘密信息 
                            num_emD = num_emD + 3;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^5) + value; %替换3位MSB
                        else
                            t = num_D - num_emD; %剩余秘密信息个数
                            bin2_8(1:t) = Encrypt_D(num_emD+1:num_emD+t); %tbit秘密信息
                            num_emD = num_emD + t;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^(8-t)) + value; %替换t位MSB
                        end 
                    end
                end   
            %------表示这个像素点可以嵌入 4 bit信息------%    
            elseif Map_origin_I(i,j) == 3  %Map=3表示原始像素值的第4MSB与其预测值相反
                bin2_8 = zeros(1,8); %用来记录要嵌入的信息，少于8位的低位(LSB)默认为0
                if num_side < num_S %辅助信息没有嵌完
                    if num_side+4 <= num_S %4位MSB都用来嵌入辅助信息
                        bin2_8(1:4) = Side_Information(num_side+1:num_side+4); 
                        num_side = num_side + 4;
                        [value] = Binary_Decimalism(bin2_8);
                        stego_I(i,j) = mod(stego_I(i,j),2^4) + value; %替换4位MSB
                    else
                        t = num_S - num_side; %剩余辅助信息个数
                        bin2_8(1:t) = Side_Information(num_side+1:num_S); %tbit辅助信息
                        num_side = num_side + t;
                        bin2_8(t+1:4) = Refer_Value(num_re+1:num_re+4-t); %(4-t)bit参考像素二进制序列信息
                        num_re = num_re + 4-t;
                        [value] = Binary_Decimalism(bin2_8);
                        stego_I(i,j) = mod(stego_I(i,j),2^4) + value; %替换4位MSB
                    end
                else
                    if num_re < num_RV  %参考像素二进制序列信息没有嵌完
                        if num_re+4 <= num_RV %4位MSB都用来嵌入参考像素二进制序列信息
                            bin2_8(1:4) = Refer_Value(num_re+1:num_re+4); 
                            num_re = num_re + 4;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^4) + value; %替换4位MSB
                        else
                            t = num_RV - num_re; %剩余参考像素二进制序列信息个数
                            bin2_8(1:t) = Refer_Value(num_re+1:num_RV); %tbit参考像素二进制序列信息
                            num_re = num_re + t;
                            bin2_8(t+1:4) = Encrypt_D(num_emD+1:num_emD+4-t); %(4-t)bit秘密信息
                            num_emD = num_emD + 4-t;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^4) + value; %替换4位MSB
                        end 
                    else
                        if num_emD+4 <= num_D
                            bin2_8(1:4) = Encrypt_D(num_emD+1:num_emD+4); %4bit秘密信息 
                            num_emD = num_emD + 4;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^4) + value; %替换4位MSB
                        else
                            t = num_D - num_emD; %剩余秘密信息个数
                            bin2_8(1:t) = Encrypt_D(num_emD+1:num_emD+t); %tbit秘密信息
                            num_emD = num_emD + t;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^(8-t)) + value; %替换t位MSB
                        end 
                    end
                end    
            %------表示这个像素点可以嵌入 5 bit信息------%    
            elseif Map_origin_I(i,j) == 4 %Map=4表示原始像素值的第5MSB与其预测值相反
                bin2_8 = zeros(1,8); %用来记录要嵌入的信息，少于8位的低位(LSB)默认为0
                if num_side < num_S %辅助信息没有嵌完
                    if num_side+5 <= num_S %5位MSB都用来嵌入辅助信息
                        bin2_8(1:5) = Side_Information(num_side+1:num_side+5); 
                        num_side = num_side + 5;
                        [value] = Binary_Decimalism(bin2_8);
                        stego_I(i,j) = mod(stego_I(i,j),2^3) + value; %替换5位MSB
                    else
                        t = num_S - num_side; %剩余辅助信息个数
                        bin2_8(1:t) = Side_Information(num_side+1:num_S); %tbit辅助信息
                        num_side = num_side + t;
                        bin2_8(t+1:5) = Refer_Value(num_re+1:num_re+5-t); %(5-t)bit参考像素二进制序列信息
                        num_re = num_re + 5-t;
                        [value] = Binary_Decimalism(bin2_8);
                        stego_I(i,j) = mod(stego_I(i,j),2^3) + value; %替换5位MSB
                    end
                else
                    if num_re < num_RV  %参考像素二进制序列信息没有嵌完
                        if num_re+5 <= num_RV %5位MSB都用来嵌入参考像素二进制序列信息
                            bin2_8(1:5) = Refer_Value(num_re+1:num_re+5); 
                            num_re = num_re + 5;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^3) + value; %替换5位MSB
                        else
                            t = num_RV - num_re; %剩余参考像素二进制序列信息个数
                            bin2_8(1:t) = Refer_Value(num_re+1:num_RV); %tbit参考像素二进制序列信息
                            num_re = num_re + t;
                            bin2_8(t+1:5) = Encrypt_D(num_emD+1:num_emD+5-t); %(5-t)bit秘密信息
                            num_emD = num_emD + 5-t;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^3) + value; %替换5位MSB
                        end 
                    else
                        if num_emD+5 <= num_D
                            bin2_8(1:5) = Encrypt_D(num_emD+1:num_emD+5); %5bit秘密信息 
                            num_emD = num_emD + 5;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^3) + value; %替换5位MSB
                        else
                            t = num_D - num_emD; %剩余秘密信息个数
                            bin2_8(1:t) = Encrypt_D(num_emD+1:num_emD+t); %tbit秘密信息
                            num_emD = num_emD + t;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^(8-t)) + value; %替换t位MSB
                        end 
                    end
                end           
            %------表示这个像素点可以嵌入 6 bit信息------%    
            elseif Map_origin_I(i,j) == 5  %Map=5表示原始像素值的第6MSB与其预测值相反
                bin2_8 = zeros(1,8); %用来记录要嵌入的信息，少于8位的低位(LSB)默认为0
                if num_side < num_S %辅助信息没有嵌完
                    if num_side+6 <= num_S %6位MSB都用来嵌入辅助信息
                        bin2_8(1:6) = Side_Information(num_side+1:num_side+6); 
                        num_side = num_side + 6;
                        [value] = Binary_Decimalism(bin2_8);
                        stego_I(i,j) = mod(stego_I(i,j),2^2) + value; %替换6位MSB
                    else
                        t = num_S - num_side; %剩余辅助信息个数
                        bin2_8(1:t) = Side_Information(num_side+1:num_S); %tbit辅助信息
                        num_side = num_side + t;
                        bin2_8(t+1:6) = Refer_Value(num_re+1:num_re+6-t); %(6-t)bit参考像素二进制序列信息
                        num_re = num_re + 6-t;
                        [value] = Binary_Decimalism(bin2_8);
                        stego_I(i,j) = mod(stego_I(i,j),2^2) + value; %替换6位MSB
                    end
                else
                    if num_re < num_RV  %参考像素二进制序列信息没有嵌完
                        if num_re+6 <= num_RV %3位MSB都用来嵌入参考像素二进制序列信息
                            bin2_8(1:6) = Refer_Value(num_re+1:num_re+6); 
                            num_re = num_re + 6;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^2) + value; %替换6位MSB
                        else
                            t = num_RV - num_re; %剩余参考像素二进制序列信息个数
                            bin2_8(1:t) = Refer_Value(num_re+1:num_RV); %tbit参考像素二进制序列信息
                            num_re = num_re + t;
                            bin2_8(t+1:6) = Encrypt_D(num_emD+1:num_emD+6-t); %(6-t)bit秘密信息
                            num_emD = num_emD + 6-t;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^2) + value; %替换6位MSB
                        end 
                    else
                        if num_emD+6 <= num_D
                            bin2_8(1:6) = Encrypt_D(num_emD+1:num_emD+6); %6bit秘密信息 
                            num_emD = num_emD + 6;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^2) + value; %替换6位MSB
                        else
                            t = num_D - num_emD; %剩余秘密信息个数
                            bin2_8(1:t) = Encrypt_D(num_emD+1:num_emD+t); %tbit秘密信息
                            num_emD = num_emD + t;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^(8-t)) + value; %替换t位MSB
                        end 
                    end
                end      
            %------表示这个像素点可以嵌入 7 bit信息------%    
            elseif Map_origin_I(i,j) == 6  %Map=6表示原始像素值的第7MSB与其预测值相反
                bin2_8 = zeros(1,8); %用来记录要嵌入的信息，少于8位的低位(LSB)默认为0
                if num_side < num_S %辅助信息没有嵌完
                    if num_side+7 <= num_S %7位MSB都用来嵌入辅助信息
                        bin2_8(1:7) = Side_Information(num_side+1:num_side+7); 
                        num_side = num_side + 7;
                        [value] = Binary_Decimalism(bin2_8);
                        stego_I(i,j) = mod(stego_I(i,j),2^1) + value; %替换7位MSB
                    else
                        t = num_S - num_side; %剩余辅助信息个数
                        bin2_8(1:t) = Side_Information(num_side+1:num_S); %tbit辅助信息
                        num_side = num_side + t;
                        bin2_8(t+1:7) = Refer_Value(num_re+1:num_re+7-t); %(7-t)bit参考像素二进制序列信息
                        num_re = num_re + 7-t;
                        [value] = Binary_Decimalism(bin2_8);
                        stego_I(i,j) = mod(stego_I(i,j),2^1) + value; %替换7位MSB
                    end
                else
                    if num_re < num_RV  %参考像素二进制序列信息没有嵌完
                        if num_re+7 <= num_RV %7位MSB都用来嵌入参考像素二进制序列信息
                            bin2_8(1:7) = Refer_Value(num_re+1:num_re+7); 
                            num_re = num_re + 7;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^1) + value; %替换7位MSB
                        else
                            t = num_RV - num_re; %剩余参考像素二进制序列信息个数
                            bin2_8(1:t) = Refer_Value(num_re+1:num_RV); %tbit参考像素二进制序列信息
                            num_re = num_re + t;
                            bin2_8(t+1:7) = Encrypt_D(num_emD+1:num_emD+7-t); %(7-t)bit秘密信息
                            num_emD = num_emD + 7-t;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^1) + value; %替换7位MSB
                        end 
                    else
                        if num_emD+7 <= num_D
                            bin2_8(1:7) = Encrypt_D(num_emD+1:num_emD+7); %7bit秘密信息 
                            num_emD = num_emD + 7;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^1) + value; %替换7位MSB
                        else
                            t = num_D - num_emD; %剩余秘密信息个数
                            bin2_8(1:t) = Encrypt_D(num_emD+1:num_emD+t); %tbit秘密信息
                            num_emD = num_emD + t;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^(8-t)) + value; %替换t位MSB
                        end 
                    end
                end           
            %------表示这个像素点可以嵌入 8 bit信息------%    
            elseif Map_origin_I(i,j) == 7 || Map_origin_I(i,j) == 8  %Map=7表示原始像素值的第8MSB(LSB)与其预测值相反
                bin2_8 = zeros(1,8); %用来记录要嵌入的信息，少于8位的低位(LSB)默认为0
                if num_side < num_S %辅助信息没有嵌完
                    if num_side+8 <= num_S %8位MSB都用来嵌入辅助信息
                        bin2_8(1:8) = Side_Information(num_side+1:num_side+8); 
                        num_side = num_side + 8;
                        [value] = Binary_Decimalism(bin2_8);
                        stego_I(i,j) = value; %替换8位MSB
                    else
                        t = num_S - num_side; %剩余辅助信息个数
                        bin2_8(1:t) = Side_Information(num_side+1:num_S); %tbit辅助信息
                        num_side = num_side + t;
                        bin2_8(t+1:8) = Refer_Value(num_re+1:num_re+8-t); %(8-t)bit参考像素二进制序列信息
                        num_re = num_re + 8-t;
                        [value] = Binary_Decimalism(bin2_8);
                        stego_I(i,j) = value; %替换8位MSB
                    end
                else
                    if num_re < num_RV  %参考像素二进制序列信息没有嵌完
                        if num_re+8 <= num_RV %8位MSB都用来嵌入参考像素二进制序列信息
                            bin2_8(1:8) = Refer_Value(num_re+1:num_re+8); 
                            num_re = num_re + 8;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = value; %替换8位MSB
                        else
                            t = num_RV - num_re; %剩余参考像素二进制序列信息个数
                            bin2_8(1:t) = Refer_Value(num_re+1:num_RV); %tbit参考像素二进制序列信息
                            num_re = num_re + t;
                            bin2_8(t+1:8) = Encrypt_D(num_emD+1:num_emD+8-t); %(8-t)bit秘密信息
                            num_emD = num_emD + 8-t;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = value; %替换8位MSB
                        end 
                    else
                        if num_emD+8 <= num_D
                            bin2_8(1:8) = Encrypt_D(num_emD+1:num_emD+8); %8bit秘密信息 
                            num_emD = num_emD + 8;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = value; %替换8位MSB
                        else
                            t = num_D - num_emD; %剩余秘密信息个数
                            bin2_8(1:t) = Encrypt_D(num_emD+1:num_emD+t); %tbit秘密信息
                            num_emD = num_emD + t;
                            [value] = Binary_Decimalism(bin2_8);
                            stego_I(i,j) = mod(stego_I(i,j),2^(8-t)) + value; %替换t位MSB
                        end 
                    end
                end         
            end
        end
    end
    % 统计嵌入的秘密数据
    emD = D(1:num_emD);
end

% ======================信息嵌入阶段======================

%% ==================对原图像的加密和嵌入全过程=====================
function [encrypt_I,stego_I,emD] = Encrypt_Embed(origin_I,D,Image_key,Data_key,ref_x,ref_y)
    % 函数说明：将原始图像origin_I加密并嵌入数据
    % 输入：origin_I（原始图像）,D（要嵌入的数据）,Image_key,Data_key（密钥）,ref_x,ref_y（参考像素的行列数）
    % 输出：encrypt_I（加密图像）,stego_I（加密标记图像）,emD（嵌入的数据）
    
    % 计算origin_I的预测值
    [origin_PV_I] = Predictor_Value(origin_I,ref_x,ref_y); 
    % 对每个像素值进行标记（即原始图像的位置图）
    [Map_origin_I] = Category_Mark(origin_PV_I,origin_I,ref_x,ref_y);
    % 将像素值的标记类别进行Huffman编码标记
    hist_Map_origin_I = tabulate(Map_origin_I(:)); %统计每个标记类别的像素值个数
    num_Map_origin_I = zeros(9,2);
    for i=1:9  % 9种类别的标记
        num_Map_origin_I(i,1) = i-1;   % num_Map_origin_I=[0 0;1 0;2 0;3 0;4 0;5 0;6 0;7 0;8 0]
    end
    [m,~] = size(hist_Map_origin_I);
    for i=1:9
        for j=2:m %hist_Map_origin_I第一行统计的是参考像素的个数
            if num_Map_origin_I(i,1) == hist_Map_origin_I(j,1)
                num_Map_origin_I(i,2) = hist_Map_origin_I(j,2);  %去掉参考像素信息，只统计标记类别信息
            end
        end
    end
    [Code,Code_Bin] = Huffman_Code(num_Map_origin_I); %计算标记的映射关系及其二进制序列表示
    % 将位置图Map_origin_I转换成二进制数组
    [Map_Bin] = Map_Binary(Map_origin_I,Code);
    % 计算存储Map_Binary长度需要的信息长度
    [row,col]=size(origin_I); 
    max = ceil(log2(row)) + ceil(log2(col)) + 2; %用这么长的二进制表示Map_Binary的长度 ceil()与floor相对，表示向上取整
    length_Map = length(Map_Bin);
    len_Bin = dec2bin(length_Map)-'0'; %将length_Map转换成二进制数组
    if length(len_Bin) < max
        len = length(len_Bin);
        B = len_Bin;
        len_Bin = zeros(1,max);
        for i=1:len
            len_Bin(max-len+i) = B(i); %存储Map_Bin的长度信息
        end 
    end
    % 统计恢复时需要的辅助信息（Code_Bin，len_Bin，Map_Bin）
    Side_Information = [Code_Bin,len_Bin,Map_Bin];
    % 对原始图像origin_I进行加密
    [encrypt_I] = Encrypt_Image(origin_I,Image_key);
    % 在Encrypt_I中嵌入信息
    [stego_I,emD] = Embed_Data(encrypt_I,Map_origin_I,Side_Information,D,Data_key,ref_x,ref_y);

end

% ==================对原图像的加密和嵌入全过程=====================：

%% ======================信息提取阶段======================
% 提取秘密信息：
function [Side_Information,Refer_Value,Encrypt_exD,Map_I,sign] = Extract_Data(stego_I,num,ref_x,ref_y)
    % 函数说明：在加密标记图像中提取信息
    % 输入：stego_I（加密标记图像）,num（秘密信息的长度）,ref_x,ref_y（参考像素的行列数）
    % 输出：Side_Information（辅助信息）,Refer_Value（参考像素信息）,Encrypt_exD（加密的秘密信息）,Map_I（位置图）,sign（判断标记）
    [row,col]=size(stego_I); %统计stego_I的行列数
    % 构建存储位置图的矩阵
    Map_I = zeros(row,col); %构建存储位置图的矩阵
    for i=1:row
        for j=1:ref_y
            Map_I(i,j) = -1; %前面ref_y列为参考像素，不进行标记
        end
    end
    for i=1:ref_x
        for j=ref_y+1:col       
            Map_I(i,j) = -1; %前面ref_x行为参考像素，不进行标记   
        end
    end
    % 先提取前ref_y列、前ref_x行中的辅助信息
    Side_Information = zeros();
    num_side = 0;%计数，统计提取辅助信息的个数
    for i=1:row
        for j=1:ref_y
            value = stego_I(i,j);
            [bin2_8] = Decimalism_Binary(value); %将十进制整数转换成8位二进制数组
            Side_Information(num_side+1:num_side+8) = bin2_8;
            num_side = num_side + 8;  
        end
    end
    for i=1:ref_x
        for j=ref_y+1:col
            value = stego_I(i,j);
            [bin2_8] = Decimalism_Binary(value); %将十进制整数转换成8位二进制数组
            Side_Information(num_side+1:num_side+8) = bin2_8;
            num_side = num_side + 8; 
        end
    end
    % 提取代表映射规则的辅助信息
    Code_Bin = Side_Information(1:32); %前32位是映射规则信息
    Code = [0,-1;1,-1;2,-1;3,-1;4,-1;5,-1;6,-1;7,-1;8,-1];
    this_end = 0;
    for i=1:9 %将二进制序列映射转换成整数映射
        last_end = this_end;
        [code_value,this_end] = Huffman_DeCode(Code_Bin,last_end);
        Code(i,2) = code_value;
    end
    % 提取位置图二进制序列的长度信息
    max = ceil(log2(row)) + ceil(log2(col)) + 2; %用这么长的二进制表示Map_I转化成二进制数列的长度
    len_Bin = Side_Information(33:32+max); %前33到32+max位是位置图二进制序列的长度信息
    num_Map = 0; %将二进制序列len_Bin转换成十进制数
    for i=1:max
        num_Map = num_Map + len_Bin(i)*(2^(max-i));
    end
    % 辅助量
    num_S = 32 + max + num_Map; %辅助信息长度
    Refer_Value = zeros();
    num_RV = (ref_x*row+ref_y*col-ref_x*ref_y)*8; %参考像素二进制序列信息的长度
    num_re = 0; %计数，统计提取参考像素二进制序列信息的长度
    Encrypt_exD = zeros();
    num_D = num; %二进制秘密信息的长度
    num_exD = 0; %计数，统计嵌入秘密信息的个数
    % 在前多行多列之外的位置提取信息
    this_end = 32 + max; %前面的辅助信息不是位置图
    sign = 1; %表示可以完全提取数据恢复图像
    for i=ref_x+1:row
        if sign == 0 %表示不能完全提取数据恢复图像
            break;
        end
        for j=ref_y+1:col
            if num_exD >= num_D %秘密数据已提取完毕
                break;
            end
            %------将当前十进制像素值转换成8位二进制数组------%
            value = stego_I(i,j); 
            [bin2_8] = Decimalism_Binary(value); 
            %--通过辅助信息计算当前像素点能提取多少bit的信息--%
            last_end = this_end;
            [map_value,this_end] = Huffman_DeCode(Side_Information,last_end);
            if map_value == -1 %表示辅助信息长度不够，无法恢复下一个Huffman编码
                sign = 0;
                break; 
            end
            for k=1:9
                if map_value == Code(k,2)
                    Map_I(i,j) = Code(k,1); %当前像素的位置图信息
                    break;
                end
            end
            %--------表示这个像素点可以提取 1 bit信息--------%
            if Map_I(i,j) == 0  %Map=0表示原始像素值的第1MSB与其预测值相反
                if num_side < num_S %辅助信息没有提取完毕
                    num_side = num_side + 1;
                    Side_Information(num_side) = bin2_8(1);
                else
                    if num_re < num_RV %参考像素二进制序列信息没有提取完毕
                        num_re = num_re + 1;
                        Refer_Value(num_re) = bin2_8(1);
                    else %最后提取秘密信息
                        num_exD = num_exD + 1;
                        Encrypt_exD(num_exD) = bin2_8(1);
                    end
                end
            %--------表示这个像素点可以提取 2 bit信息--------%
            elseif Map_I(i,j) == 1 %Map=1表示原始像素值的第2MSB与其预测值相反
                if num_side < num_S %辅助信息没有提取完毕
                    if num_side+2 <= num_S %2位MSB都是辅助信息
                        Side_Information(num_side+1:num_side+2) = bin2_8(1:2);
                        num_side = num_side + 2;
                    else
                        num_side = num_side + 1; %1bit辅助信息
                        Side_Information(num_side) = bin2_8(1);
                        num_re = num_re + 1; %1bit参考像素二进制序列信息
                        Refer_Value(num_re) = bin2_8(2);                   
                    end
                else
                    if num_re < num_RV %参考像素二进制序列信息没有提取完毕
                        if num_re+2 <= num_RV %2位MSB都是参考像素二进制序列信息
                            Refer_Value(num_re+1:num_re+2) = bin2_8(1:2);
                            num_re = num_re + 2;
                        else
                            num_re = num_re + 1; %1bit参考像素二进制序列信息
                            Refer_Value(num_re) = bin2_8(1);  
                            num_exD = num_exD + 1; %1bit秘密信息
                            Encrypt_exD(num_exD) = bin2_8(2);
                        end
                    else
                        if num_exD+2 <= num_D
                            Encrypt_exD(num_exD+1:num_exD+2) = bin2_8(1:2); %2bit秘密信息
                            num_exD = num_exD + 2;
                        else
                            num_exD = num_exD + 1; %1bit秘密信息
                            Encrypt_exD(num_exD) = bin2_8(1);
                        end
                    end
                end 
            %--------表示这个像素点可以提取 3 bit信息--------%
            elseif Map_I(i,j) == 2  %Map=2表示原始像素值的第3MSB与其预测值相反
                if num_side < num_S %辅助信息没有提取完毕
                    if num_side+3 <= num_S %3位MSB都是辅助信息
                        Side_Information(num_side+1:num_side+3) = bin2_8(1:3);
                        num_side = num_side + 3;
                    else
                        t = num_S - num_side; %剩余辅助信息个数
                        Side_Information(num_side+1:num_side+t) = bin2_8(1:t); %tbit辅助信息
                        num_side = num_side + t;
                        Refer_Value(num_re+1:num_re+3-t) = bin2_8(t+1:3); %(3-t)bit参考像素二进制序列信息
                        num_re = num_re + 3-t;                 
                    end
                else
                    if num_re < num_RV %参考像素二进制序列信息没有提取完毕
                        if num_re+3 <= num_RV %3位MSB都是参考像素二进制序列信息
                            Refer_Value(num_re+1:num_re+3) = bin2_8(1:3);
                            num_re = num_re + 3;
                        else
                            t = num_RV - num_re; %剩余参考像素二进制序列信息个数
                            Refer_Value(num_re+1:num_re+t) = bin2_8(1:t); %tbit参考像素二进制序列信息
                            num_re = num_re + t;
                            Encrypt_exD(num_exD+1:num_exD+3-t) = bin2_8(t+1:3); %(3-t)bit秘密信息
                            num_exD = num_exD + 3-t;
                        end
                    else
                        if num_exD+3 <= num_D
                            Encrypt_exD(num_exD+1:num_exD+3) = bin2_8(1:3); %3bit秘密信息
                            num_exD = num_exD + 3;
                        else
                            t = num_D - num_exD;
                            Encrypt_exD(num_exD+1:num_exD+t) = bin2_8(1:t); %tbit秘密信息
                            num_exD = num_exD + t; 
                        end
                    end
                end
            %--------表示这个像素点可以提取 4 bit信息--------%
            elseif Map_I(i,j) == 3  %Map=3表示原始像素值的第4MSB与其预测值相反
                if num_side < num_S %辅助信息没有提取完毕
                    if num_side+4 <= num_S %4位MSB都是辅助信息
                        Side_Information(num_side+1:num_side+4) = bin2_8(1:4);
                        num_side = num_side + 4;
                    else
                        t = num_S - num_side; %剩余辅助信息个数
                        Side_Information(num_side+1:num_side+t) = bin2_8(1:t); %tbit辅助信息
                        num_side = num_side + t;
                        Refer_Value(num_re+1:num_re+4-t) = bin2_8(t+1:4); %(4-t)bit参考像素二进制序列信息
                        num_re = num_re + 4-t;                 
                    end
                else
                    if num_re < num_RV %参考像素二进制序列信息没有提取完毕
                        if num_re+4 <= num_RV %4位MSB都是参考像素二进制序列信息
                            Refer_Value(num_re+1:num_re+4) = bin2_8(1:4);
                            num_re = num_re + 4;
                        else
                            t = num_RV - num_re; %剩余参考像素二进制序列信息个数
                            Refer_Value(num_re+1:num_re+t) = bin2_8(1:t); %tbit参考像素二进制序列信息
                            num_re = num_re + t;
                            Encrypt_exD(num_exD+1:num_exD+4-t) = bin2_8(t+1:4); %(4-t)bit秘密信息
                            num_exD = num_exD + 4-t;
                        end
                    else
                        if num_exD+4 <= num_D
                            Encrypt_exD(num_exD+1:num_exD+4) = bin2_8(1:4); %4bit秘密信息
                            num_exD = num_exD + 4;
                        else
                            t = num_D - num_exD;
                            Encrypt_exD(num_exD+1:num_exD+t) = bin2_8(1:t); %tbit秘密信息
                            num_exD = num_exD + t; 
                        end
                    end
                end
            %--------表示这个像素点可以提取 5 bit信息--------%
            elseif Map_I(i,j) == 4  %Map=4表示原始像素值的第5MSB与其预测值相反
                if num_side < num_S %辅助信息没有提取完毕
                    if num_side+5 <= num_S %5位MSB都是辅助信息
                        Side_Information(num_side+1:num_side+5) = bin2_8(1:5);
                        num_side = num_side + 5;
                    else
                        t = num_S - num_side; %剩余辅助信息个数
                        Side_Information(num_side+1:num_side+t) = bin2_8(1:t); %tbit辅助信息
                        num_side = num_side + t;
                        Refer_Value(num_re+1:num_re+5-t) = bin2_8(t+1:5); %(5-t)bit参考像素二进制序列信息
                        num_re = num_re + 5-t;                 
                    end
                else
                    if num_re < num_RV %参考像素二进制序列信息没有提取完毕
                        if num_re+5 <= num_RV %5位MSB都是参考像素二进制序列信息
                            Refer_Value(num_re+1:num_re+5) = bin2_8(1:5);
                            num_re = num_re + 5;
                        else
                            t = num_RV - num_re; %剩余参考像素二进制序列信息个数
                            Refer_Value(num_re+1:num_re+t) = bin2_8(1:t); %tbit参考像素二进制序列信息
                            num_re = num_re + t;
                            Encrypt_exD(num_exD+1:num_exD+5-t) = bin2_8(t+1:5); %(5-t)bit秘密信息
                            num_exD = num_exD + 5-t;
                        end
                    else
                        if num_exD+5 <= num_D
                            Encrypt_exD(num_exD+1:num_exD+5) = bin2_8(1:5); %5bit秘密信息
                            num_exD = num_exD + 5;
                        else
                            t = num_D - num_exD;
                            Encrypt_exD(num_exD+1:num_exD+t) = bin2_8(1:t); %tbit秘密信息
                            num_exD = num_exD + t; 
                        end
                    end
                end
                %--------表示这个像素点可以提取 6 bit信息--------%
            elseif Map_I(i,j) == 5  %Map=5表示原始像素值的第6MSB与其预测值相反
                if num_side < num_S %辅助信息没有提取完毕
                    if num_side+6 <= num_S %6位MSB都是辅助信息
                        Side_Information(num_side+1:num_side+6) = bin2_8(1:6);
                        num_side = num_side + 6;
                    else
                        t = num_S - num_side; %剩余辅助信息个数
                        Side_Information(num_side+1:num_side+t) = bin2_8(1:t); %tbit辅助信息
                        num_side = num_side + t;
                        Refer_Value(num_re+1:num_re+6-t) = bin2_8(t+1:6); %(6-t)bit参考像素二进制序列信息
                        num_re = num_re + 6-t;                 
                    end
                else
                    if num_re < num_RV %参考像素二进制序列信息没有提取完毕
                        if num_re+6 <= num_RV %6位MSB都是参考像素二进制序列信息
                            Refer_Value(num_re+1:num_re+6) = bin2_8(1:6);
                            num_re = num_re + 6;
                        else
                            t = num_RV - num_re; %剩余参考像素二进制序列信息个数
                            Refer_Value(num_re+1:num_re+t) = bin2_8(1:t); %tbit参考像素二进制序列信息
                            num_re = num_re + t;
                            Encrypt_exD(num_exD+1:num_exD+6-t) = bin2_8(t+1:6); %(6-t)bit秘密信息
                            num_exD = num_exD + 6-t;
                        end
                    else
                        if num_exD+6 <= num_D
                            Encrypt_exD(num_exD+1:num_exD+6) = bin2_8(1:6); %6bit秘密信息
                            num_exD = num_exD + 6;
                        else
                            t = num_D - num_exD;
                            Encrypt_exD(num_exD+1:num_exD+t) = bin2_8(1:t); %tbit秘密信息
                            num_exD = num_exD + t; 
                        end
                    end
                end
                %--------表示这个像素点可以提取 7 bit信息--------%
            elseif Map_I(i,j) == 6  %Map=6表示原始像素值的第7MSB与其预测值相反
                if num_side < num_S %辅助信息没有提取完毕
                    if num_side+7 <= num_S %7位MSB都是辅助信息
                        Side_Information(num_side+1:num_side+7) = bin2_8(1:7);
                        num_side = num_side + 7;
                    else
                        t = num_S - num_side; %剩余辅助信息个数
                        Side_Information(num_side+1:num_side+t) = bin2_8(1:t); %tbit辅助信息
                        num_side = num_side + t;
                        Refer_Value(num_re+1:num_re+7-t) = bin2_8(t+1:7); %(7-t)bit参考像素二进制序列信息
                        num_re = num_re + 7-t;                 
                    end
                else
                    if num_re < num_RV %参考像素二进制序列信息没有提取完毕
                        if num_re+7 <= num_RV %7位MSB都是参考像素二进制序列信息
                            Refer_Value(num_re+1:num_re+7) = bin2_8(1:7);
                            num_re = num_re + 7;
                        else
                            t = num_RV - num_re; %剩余参考像素二进制序列信息个数
                            Refer_Value(num_re+1:num_re+t) = bin2_8(1:t); %tbit参考像素二进制序列信息
                            num_re = num_re + t;
                            Encrypt_exD(num_exD+1:num_exD+7-t) = bin2_8(t+1:7); %(7-t)bit秘密信息
                            num_exD = num_exD + 7-t;
                        end
                    else
                        if num_exD+7 <= num_D
                            Encrypt_exD(num_exD+1:num_exD+7) = bin2_8(1:7); %7bit秘密信息
                            num_exD = num_exD + 7;
                        else
                            t = num_D - num_exD;
                            Encrypt_exD(num_exD+1:num_exD+t) = bin2_8(1:t); %tbit秘密信息
                            num_exD = num_exD + t; 
                        end
                    end
                end
                %--------表示这个像素点可以提取 8 bit信息--------%
            elseif Map_I(i,j) == 7 || Map_I(i,j) == 8  %Map=7表示原始像素值的第8MSB(LSB)与其预测值相反
                if num_side < num_S %辅助信息没有提取完毕
                    if num_side+8 <= num_S %8位MSB都是辅助信息
                        Side_Information(num_side+1:num_side+8) = bin2_8(1:8);
                        num_side = num_side + 8;
                    else
                        t = num_S - num_side; %剩余辅助信息个数
                        Side_Information(num_side+1:num_side+t) = bin2_8(1:t); %tbit辅助信息
                        num_side = num_side + t;
                        Refer_Value(num_re+1:num_re+8-t) = bin2_8(t+1:8); %(8-t)bit参考像素二进制序列信息
                        num_re = num_re + 8-t;                 
                    end
                else
                    if num_re < num_RV %参考像素二进制序列信息没有提取完毕
                        if num_re+8 <= num_RV %8位MSB都是参考像素二进制序列信息
                            Refer_Value(num_re+1:num_re+8) = bin2_8(1:8);
                            num_re = num_re + 8;
                        else
                            t = num_RV - num_re; %剩余参考像素二进制序列信息个数
                            Refer_Value(num_re+1:num_re+t) = bin2_8(1:t); %tbit参考像素二进制序列信息
                            num_re = num_re + t;
                            Encrypt_exD(num_exD+1:num_exD+8-t) = bin2_8(t+1:8); %(8-t)bit秘密信息
                            num_exD = num_exD + 8-t;
                        end
                    else
                        if num_exD+8 <= num_D
                            Encrypt_exD(num_exD+1:num_exD+8) = bin2_8(1:8); %8bit秘密信息
                            num_exD = num_exD + 8;
                        else
                            t = num_D - num_exD;
                            Encrypt_exD(num_exD+1:num_exD+t) = bin2_8(1:t); %tbit秘密信息
                            num_exD = num_exD + t; 
                        end
                    end
                end 
            end
        end
    end
end

% ======================信息提取阶段======================

%% ======================图像恢复阶段======================
function [recover_I] = Recover_Image(stego_I,Image_key,Side_Information,Refer_Value,Map_I,num,ref_x,ref_y)
    % 函数说明：根据提取的辅助信息恢复图像
    % 输入：stego_I（载密图像）,Image_key（图像加密密钥）,Side_Information（辅助信息）,Refer_Value（参考像素信息）,Map_I（位置图）,num（秘密信息的长度）,ref_x,ref_y（参考像素的行列数）
    % 输出：recover_I（恢复图像）
    [row,col] = size(stego_I); %统计stego_I的行列数
    % 根据Refer_Value恢复前ref_y列、前ref_x行的参考像素
    refer_I = stego_I;
    t = 0; %计数
    for i=1:row
        for j=1:ref_y
            bin2_8 = Refer_Value(t+1:t+8);
            [value] = Binary_Decimalism(bin2_8); %将8位二进制数组转换成十进制整数
            refer_I(i,j) = value;
            t = t + 8;
        end
    end
    for i=1:ref_x
        for j=ref_y+1:col
            bin2_8 = Refer_Value(t+1:t+8);
            [value] = Binary_Decimalism(bin2_8); %将8位二进制数组转换成十进制整数
            refer_I(i,j) = value;
            t = t + 8;
        end
    end
    % 将图像refer_I根据图像加密密钥解密
    [decrypt_I] = Encrypt_Image(refer_I,Image_key);
    % 根据Side_Information、Map_I和num恢复其他位置的像素
    recover_I = decrypt_I;
    num_S = length(Side_Information);
    num_D = num_S + num; %嵌入信息的总数
    re = 0; %计数
    for i=ref_x+1:row
        for j=ref_y+1:col
            if re >= num_D %嵌入信息的比特位全部恢复完毕
                break;
            end
            %---------求当前像素点的预测值---------%
            a = recover_I(i-1,j);
            b = recover_I(i-1,j-1);
            c = recover_I(i,j-1);
            if b <= min(a,c)
                pv = max(a,c);
            elseif b >= max(a,c)
                pv = min(a,c);
            else
                pv = a + c - b;
            end
            %--将原始值和预测值转换成8位二进制数组--%
            x = recover_I(i,j);
            [bin2_x] = Decimalism_Binary(x);
            [bin2_pv] = Decimalism_Binary(pv);
            %--------表示这个像素点需要恢复 1 bit MSB--------%
            if Map_I(i,j) == 0  %Map=0表示原始像素值的第1MSB与其预测值相反
                if bin2_pv(1) == 0 
                    bin2_x(1) = 1; 
                else  
                    bin2_x(1) = 0;
                end
                [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                recover_I(i,j) = value;
                re = re + 1; %恢复1bit  
            %--------表示这个像素点需要恢复 2 bit MSB--------%
            elseif Map_I(i,j) == 1  %Map=1表示原始像素值的第2MSB与其预测值相反
                if re+2 <= num_D
                    if bin2_pv(2) == 0
                        bin2_x(2) = 1;
                    else
                        bin2_x(2) = 0;
                    end
                    bin2_x(1) = bin2_pv(1);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + 2; %恢复2bit
                else 
                    t = num_D - re; %剩余恢复的bit数
                    bin2_x(1:t) = bin2_pv(1:t);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + t; %恢复tbit 
                end
            %--------表示这个像素点需要恢复 3 bit MSB--------%
            elseif Map_I(i,j) == 2  %Map=2表示原始像素值的第3MSB与其预测值相反
                if re+2 <= num_D
                    if bin2_pv(3) == 0 
                        bin2_x(3) = 1; 
                    else                    
                        bin2_x(3) = 0;
                    end
                    bin2_x(1:2) = bin2_pv(1:2);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + 3; %恢复3bit
                else 
                    t = num_D - re; %剩余恢复的bit数
                    bin2_x(1:t) = bin2_pv(1:t);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + t; %恢复tbit
                end    
            %--------表示这个像素点需要恢复 4 bit MSB--------%
            elseif Map_I(i,j) == 3  %Map=3表示原始像素值的第4MSB与其预测值相反
                if re+3 <= num_D
                    if bin2_pv(4) == 0 
                        bin2_x(4) = 1; 
                    else                    
                        bin2_x(4) = 0;
                    end
                    bin2_x(1:3) = bin2_pv(1:3);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + 4; %恢复4bit
                else 
                    t = num_D - re; %剩余恢复的bit数
                    bin2_x(1:t) = bin2_pv(1:t);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + t; %恢复tbit
                end  
            %--------表示这个像素点需要恢复 5 bit MSB--------%
            elseif Map_I(i,j) == 4  %Map=4表示原始像素值的第5MSB与其预测值相反
                if re+4 <= num_D
                    if bin2_pv(5) == 0 
                        bin2_x(5) = 1; 
                    else                    
                        bin2_x(5) = 0;
                    end
                    bin2_x(1:4) = bin2_pv(1:4);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + 5; %恢复5bit
                else 
                    t = num_D - re; %剩余恢复的bit数
                    bin2_x(1:t) = bin2_pv(1:t);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + t; %恢复tbit
                end    
            %--------表示这个像素点需要恢复 6 bit MSB--------%
            elseif Map_I(i,j) == 5  %Map=5表示原始像素值的第6MSB与其预测值相反
                if re+5 <= num_D
                    if bin2_pv(6) == 0 
                        bin2_x(6) = 1; 
                    else                    
                        bin2_x(6) = 0;
                    end
                    bin2_x(1:5) = bin2_pv(1:5);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + 6; %恢复6bit
                else 
                    t = num_D - re; %剩余恢复的bit数
                    bin2_x(1:t) = bin2_pv(1:t);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + t; %恢复tbit
                end 
            %--------表示这个像素点需要恢复 7 bit MSB--------%
            elseif Map_I(i,j) == 6  %Map=6表示原始像素值的第7MSB与其预测值相反
                if re+6 <= num_D
                    if bin2_pv(7) == 0 
                        bin2_x(7) = 1; 
                    else                    
                        bin2_x(7) = 0;
                    end
                    bin2_x(1:6) = bin2_pv(1:6);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + 7; %恢复7bit
                else 
                    t = num_D - re; %剩余恢复的bit数
                    bin2_x(1:t) = bin2_pv(1:t);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + t; %恢复tbit
                end 
            %--------表示这个像素点需要恢复 8 bit MSB--------%
            elseif Map_I(i,j) == 7  %Map=7表示原始像素值的第8MSB与其预测值相反
                if re+7 <= num_D
                    if bin2_pv(8) == 0 
                        bin2_x(8) = 1; 
                    else                    
                        bin2_x(8) = 0;
                    end
                    bin2_x(1:7) = bin2_pv(1:7);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + 8; %恢复8bit
                else 
                    t = num_D - re; %剩余恢复的bit数
                    bin2_x(1:t) = bin2_pv(1:t);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + t; %恢复tbit
                end  
            %--------表示这个像素点需要恢复 8 bit MSB--------%
            elseif Map_I(i,j) == 8  %Map=8表示原始像素值等于其预测值
                if re+8 <= num_D
                    bin2_x(1:8) = bin2_pv(1:8);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + 8; %恢复8bit
                else 
                    t = num_D - re; %剩余恢复的bit数
                    bin2_x(1:t) = bin2_pv(1:t);
                    [value] = Binary_Decimalism(bin2_x); %将8位二进制数组转换成十进制整数
                    recover_I(i,j) = value;
                    re = re + t; %恢复tbit
                end
            end
        end
    end
end
% ======================图像恢复阶段======================











