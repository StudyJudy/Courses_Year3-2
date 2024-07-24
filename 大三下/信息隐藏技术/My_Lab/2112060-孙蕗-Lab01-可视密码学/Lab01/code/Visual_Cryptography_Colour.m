function Visual_Cryptography_Colour()

path = input("请输入要 彩色图像的（2,2）的可视密钥分享方案 加密的图片：",'s');

colour = imread(path);
figure('name','colour');
imshow(colour);

Size=size(colour);
scenery_ht=zeros(Size);

recover3=zeros(2*Size(1),2*Size(2),Size(3));

pic1=zeros(2*Size(1),2*Size(2),Size(3));
pic2=zeros(2*Size(1),2*Size(2),Size(3));


% 将彩色图像的每个分量当做一张图片来处理，即把一张彩色图像看做红、绿、蓝三个分量上的三张图片
% 对每一张图片按照灰度图像进行HalfTone半色调处理
for i=1:3
    % im2ht()半色调化处理
    scenery_ht(:,:,i) = im2ht(colour(:,:,i));
   
    figure('name','半色调化处理');
    imshow(scenery_ht(:,:,i));
    % imwrite(origin,'ht.jpg');

    % 调用decompose()分解得到子图
    % 对每一张图片进行信息分存
    [pic1(:,:,i),pic2(:,:,i)] = decompose(scenery_ht(:,:,i));

    % 子图1
    figure('name','pic1');
    imshow(pic1(:,:,i));
    % imwrite(pic1,'dec1.jpg');

    % 子图2
    figure('name','pic2');
    imshow(pic2(:,:,i));
    % imwrite(pic2,'dec2.jpg');
    
    % R、G、B分量分别分存到两张子图中
    % 最后将得到的子图进行合并可以得到两张彩色子图
    recover3(:,:,i) = merge(pic1(:,:,i),pic2(:,:,i));

    % merge()两张图像叠加还原
    figure('name','recover3');
    imshow(recover3(:,:,i));
    % imwrite(recover3,'recover3.jpg');

end

figure('name','pic1_final');
imshow(pic1);
imwrite(pic1,'scenery_viscrypto_colour_dec_pic1.jpg');

figure('name','pic2_final');
imshow(pic2);
imwrite(pic2,'scenery_viscrypto_colour_dec_pic2.jpg');

figure('name','scenery_ht_final');
imshow(scenery_ht);
imwrite(scenery_ht,'scenery_viscrypto_colour_halftone.jpg');

figure('name','recover3_final');
imshow(recover3);
imwrite(recover3,'scenery_viscrypto_colour_rec.jpg');

end


% 使用误差扩散法(Error Diffusion)进行图像半色调处理(HalfTone)

function scenery_ht = im2ht(colour)
Size=size(colour);
x=Size(1);
y=Size(2);

% 误差扩散法的基本思想是，先阈值量化图像像素，然后将图像量化过程中产生的误差分配给周围像素点。
% 利用这种方法所生成的半色调图像能够比较忠实地反映图像的灰度层次过渡。

% 灰度图像是具有多个灰度等级层次丰富的连续调图像，每个像素点用8比特表示，取值为0～255。
for i=1:x
    for j=1:y
        % 如果输入的像素值小于等于127(比较接近黑色),则输出0(黑色);
        % 如果输出的像素值大于127(比较接近白色),则输出255(白色)
        if colour(i,j)>127
            out=255;
        else
            out=0;
        end
        
        % 误差=输入像素值-输出
        error=colour(i,j)-out;
        
        % 误差分配
        if j>1 && j<255 && i<255
            colour(i,j+1)=colour(i,j+1)+error*7/16.0;  %右方
            colour(i+1,j)=colour(i+1,j)+error*5/16.0;  %下方
            colour(i+1,j-1)=colour(i+1,j-1)+error*3/16.0;  %左下方
            colour(i+1,j+1)=colour(i+1,j+1)+error*1/16.0;  %右下方
            colour(i,j)=out;
        else
            colour(i,j)=out;
        end
    end
end
scenery_ht=colour;

end

% decompose函数将图像分解成两个子图像
function [pic1,pic2] = decompose(scenery_ht)

%初始化
Size=size(scenery_ht);
x=Size(1);
y=Size(2);

% 创建一个全为255的大小为2*x by 2*y的矩阵
pic1=zeros(2*x,2*y);
pic1(:,:)=255;

% 创建一个全为255的大小为2*x by 2*y的矩阵
pic2=zeros(2*x,2*y);
pic2(:,:)=255;

%take image1 as first
for i = 1:x
    for j = 1:y
        % 生成一个1到3之间的随机整数
        % 用于随机选择图像中每个像素所属的子图像
        key = randi(3);
        
        % 计算子图的起始横坐标
        % 根据当前像素在原始图像中的位置，计算出其在子图中的位置
        son_x=1+2*(i-1);

        % 计算子图的起始纵坐标
        % 根据当前像素在原始图像中的位置，计算出其在子图中的位置
        son_y=1+2*(j-1);

        % 根据随机数 key 的值决定当前像素属于哪个子图
        switch key
            case 1
                % 若随机数为 1，则当前像素属于子图1
                % 将子图1对应位置的像素值设为黑色
                pic1(son_x,son_y)=0; pic1(son_x,son_y+1)=0;
                % 判断原始图像中对应位置的像素颜色，若为黑色，则将子图2对应位置的像素设为黑色
                if scenery_ht(i,j)==0
                    %origin is black
                    pic2(son_x+1,son_y)=0; pic2(son_x+1,son_y+1)=0;
                else
                    %origin is white
                    pic2(son_x,son_y+1)=0; pic2(son_x+1,son_y+1)=0;
                end
            case 2
                % 若随机数为 2，则当前像素属于子图1
                % 将子图1对应位置的像素值设为黑色
                pic1(son_x,son_y)=0; pic1(son_x+1,son_y+1)=0;
                % 判断原始图像中对应位置的像素颜色，若为黑色，则将子图2对应位置的像素设为黑色
                if scenery_ht(i,j)==0
                    %origin is black
                    pic2(son_x,son_y+1)=0; pic2(son_x+1,son_y)=0;
                else
                    %origin is white
                    pic2(son_x,son_y)=0; pic2(son_x+1,son_y)=0;
                end
            case 3
                % 若随机数为 3，则当前像素属于子图1
                % 将子图1对应位置的像素值设为黑色
                pic1(son_x,son_y)=0; pic1(son_x+1,son_y)=0;
                % 判断原始图像中对应位置的像素颜色，若为黑色，则将子图2对应位置的像素设为黑色
                if scenery_ht(i,j)==0
                    %origin is black
                    pic2(son_x,son_y+1)=0; pic2(son_x+1,son_y+1)=0;
                else
                    %origin is white
                    pic2(son_x,son_y)=0; pic2(son_x,son_y+1)=0;
                end
        end
    end
end

end


function recover3 = merge(pic1,pic2)
Size=size(pic1);
x=Size(1);
y=Size(2);

% 创建一个大小为x by y的全零矩阵
recover3=zeros(x,y);
recover3(:,:)=255;

for i=1:x
    for j=1:y
        % 将两个子图像逐像素进行AND操作，得到还原后的图像
        recover3(i,j)=pic1(i,j)&pic2(i,j);
    end
end

end