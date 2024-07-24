function Visual_Cryptography_Gray()

path = input("请输入要加密的图片：",'s');

origin= imread(path);
figure('name','origin');
imshow(origin);

% 转为灰度图
image=rgb2gray(origin); 

figure('name','灰度图');
imshow(image);
imwrite(image,'snowman_viscrypto_gray.jpg');


% im2ht()半色调化处理
origin_ht = im2ht(image);
figure('name','半色调化图');
imshow(origin_ht);
imwrite(origin_ht,'snowman_viscrypto_gray_halftone.jpg');


% 调用decompose()分解得到子图
[pic1,pic2] = decompose(origin_ht);
% imshow(pic1);
% imshow(pic2);

% 子图一
figure('name','decompose分解子图1');
imshow(pic1);
imwrite(pic1,'snowman_viscrypto_gray_dec_pic1.jpg');

% 子图二
figure('name','decompose分解子图2');
imshow(pic2);
imwrite(pic2,'snowman_viscrypto_gray_dec_pic2.jpg');

% merge()两张图像叠加还原
recover2 = merge(pic1,pic2);

figure('name','叠加还原');
imshow(recover2);
imwrite(recover2,'snowman_viscrypto_gray_rec.jpg');
end



% 使用误差扩散法(Error Diffusion)进行图像半色调处理(HalfTone)
function image = im2ht(image_gray)
Size=size(image_gray);
x=Size(1);
y=Size(2);

% 误差扩散法的基本思想是，先阈值量化图像像素，然后将图像量化过程中产生的误差分配给周围像素点。
% 利用这种方法所生成的半色调图像能够比较忠实地反映图像的灰度层次过渡。

% 灰度图像是具有多个灰度等级层次丰富的连续调图像，每个像素点用8比特表示，取值为0～255。
for i=1:x
    for j=1:y
        % 如果输入的像素值小于等于127(比较接近黑色),则输出0(黑色);如果输出的像素值大于127(比较接近白色),则输出255(白色)。
        if image_gray(i,j)>127
            out=255;
        else
            out=0;
        end

        % 误差=输入像素值-输出
        error=image_gray(i,j)-out;

        % 误差分配
        if j>1 && j<255 && i<255
            image_gray(i,j+1)=image_gray(i,j+1)+error*7/16.0;  %右方
            image_gray(i+1,j)=image_gray(i+1,j)+error*5/16.0;  %下方
            image_gray(i+1,j-1)=image_gray(i+1,j-1)+error*3/16.0;  %左下方
            image_gray(i+1,j+1)=image_gray(i+1,j+1)+error*1/16.0;  %右下方
            image_gray(i,j)=out;
        else
            image_gray(i,j)=out;
        end
    end
end
image=image_gray;

end

% decompose函数将二值图像分解成两个子图像
function [pic1,pic2] = decompose(origin_ht)

%初始化
Size=size(origin_ht);
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
                if origin_ht(i,j)==0
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
                if origin_ht(i,j)==0
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
                if origin_ht(i,j)==0
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


function recover2 = merge(pic1,pic2)
Size=size(pic1);
x=Size(1);
y=Size(2);

% 创建一个大小为x by y的全零矩阵
recover2=zeros(x,y);
recover2(:,:)=255;

for i=1:x
    for j=1:y
        % 将两个子图像逐像素进行AND操作，得到还原后的图像
        recover2(i,j)=pic1(i,j)&pic2(i,j);
    end
end

end