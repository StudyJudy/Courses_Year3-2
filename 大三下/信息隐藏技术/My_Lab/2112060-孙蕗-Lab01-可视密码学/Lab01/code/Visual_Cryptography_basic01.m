function Visual_Cryptography_basic01()

% 输入一张256×256bmp图片
% 输出两张512×512bmp子图片和一张512×512bmp还原图片

path = input("请输入 二值图像的（2,2）的可视密钥分享方案 要加密的图片：",'s');
origin = imread(path);

% 二值图像
bwimg = im2bw(origin);
% bwimg = imbinarize(origin);
figure('name','二值黑白图');
imshow(bwimg);
imwrite(bwimg,'binary_image_flower.bmp');

% 调用decompose()分解得到子图
[pic1,pic2] = decompose(bwimg);
% imshow(bwimg);

% 子图一
figure('name','pic1');
imshow(pic1);
imwrite(pic1,'pic1_01_flower.bmp');

% 子图二
figure('name','pic2');
imshow(pic2);
imwrite(pic2,'pic2_01_flower.bmp');

% 两张图像叠加还原
recover1 = merge(pic1,pic2);

figure('name','叠加还原图像');
imshow(recover1);
imwrite(recover1,'recover1_flower.bmp');

end

% decompose函数将二值图像分解成两个子图像
function [pic1,pic2] = decompose(bwimg)
%初始化
Size=size(bwimg);
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
                if bwimg(i,j)==0
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
                if bwimg(i,j)==0
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
                if bwimg(i,j)==0
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

function recover1 = merge(pic1,pic2)
Size=size(pic1);
x=Size(1);
y=Size(2);

% 创建一个大小为x by y的全零矩阵
recover1=zeros(x,y);
recover1(:,:)=255;

for i=1:x
    for j=1:y
        % 将两个子图像逐像素进行AND操作，得到还原后的图像
        recover1(i,j)=pic1(i,j)&pic2(i,j);
    end
end

end