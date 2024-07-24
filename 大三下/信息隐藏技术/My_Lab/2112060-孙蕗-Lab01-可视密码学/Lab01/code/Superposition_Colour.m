function Superposition_Colour()

path = input("请输入彩色图像叠相术要加密的图片：",'s');%snowman.jpg
origin = imread(path);
% imshow(origin);
path = input("请输入彩色图像叠相术隐藏图1：",'s');%boy.jpg
origin1 = imread(path);
% imshow(origin1);
path = input("请输入彩色图像叠相术隐藏图2：",'s');%flower.jpg
origin2 = imread(path);
% imshow(origin2);

Size=size(origin);

% 对三张彩色图像进行半色调处理
origin_ht=zeros(Size);
origin1_ht=zeros(Size);
origin2_ht=zeros(Size);

image1=zeros(2*Size(1),2*Size(2),3);
image2=zeros(2*Size(1),2*Size(2),3);

originB=zeros(2*Size(1),2*Size(2),3);

% 将彩色图像的每个分量当做一张图片来处理，即把一张彩色图像看做红、绿、蓝三个分量上的三张图片
% 对每一张图片按照灰度图像进行HalfTone半色调处理
for i=1:3

% 半色调化处理
origin_ht(:,:,i) = in2ht(origin(:,:,i));
origin1_ht(:,:,i) = in2ht(origin1(:,:,i));
origin2_ht(:,:,i) = in2ht(origin2(:,:,i));

% 信息分存到两个子图
[image1(:,:,i),image2(:,:,i)] = decompose(origin_ht(:,:,i),origin1_ht(:,:,i),origin2_ht(:,:,i));

%merge叠加还原
originB(:,:,i) = merge(image1(:,:,i),image2(:,:,i));
end

%半色调化处理后的图像
imshow(origin_ht);
imwrite(origin_ht,'snowman_halftone.jpg');
imshow(origin1_ht);
imwrite(origin1_ht,'boy_halftone.jpg');
imshow(origin2_ht);
imwrite(origin2_ht,'flower_halftone.jpg');

% 分存子图1 
imshow(image1);
imwrite(image1,'colour_camouflage1.jpg');
% 分存子图2
imshow(image2);
imwrite(image2,'colour_camouflage2.jpg');

%叠加还原后图像
imshow(originB);
imwrite(originB,'colour_recovered.jpg');
end

% 使用误差扩散法(Error Diffusion)进行图像半色调处理(HalfTone)
function image = in2ht(gray)
Size=size(gray);
x=Size(1);
y=Size(2);

% 误差扩散法的基本思想是，先阈值量化图像像素，然后将图像量化过程中产生的误差分配给周围像素点。
% 利用这种方法所生成的半色调图像能够比较忠实地反映图像的灰度层次过渡。

% 灰度图像是具有多个灰度等级层次丰富的连续调图像，每个像素点用8比特表示，取值为0～255。
for m=1:x
    for n=1:y
        % 如果输入的像素值小于等于127(比较接近黑色),则输出0(黑色);
        % 如果输出的像素值大于127(比较接近白色),则输出255(白色)
        if gray(m,n)>127
            out=255;
        else
            out=0;
        end

        % 误差=输入像素值-输出
        error=gray(m,n)-out;
        % 误差分配
        if n>1 && n<255 && m<255
            gray(m,n+1)=gray(m,n+1)+error*7/16.0;  %右方
            gray(m+1,n)=gray(m+1,n)+error*5/16.0;  %下方
            gray(m+1,n-1)=gray(m+1,n-1)+error*3/16.0;  %左下方
            gray(m+1,n+1)=gray(m+1,n+1)+error*1/16.0;  %右下方
            gray(m,n)=out;
        else
            gray(m,n)=out;
        end
    end
end
image=gray;

end

% decompose函数将图像分存成两个子图像
function [image1,image2] = decompose(origin,origin1,origin2)
%初始化
Size=size(origin);
x=Size(1);
y=Size(2);

% 创建一个全为255的大小为2*x by 2*y的矩阵
image1=zeros(2*x,2*y);
image1(:,:)=255;

% 创建一个全为255的大小为2*x by 2*y的矩阵
image2=zeros(2*x,2*y);
image2(:,:)=255;

%take image1 as first
for i = 1:x
    for j = 1:y
        key = randi(4);
        son_x=1+2*(i-1);
        son_y=1+2*(j-1);
        switch key
            case 1
                if origin(i,j)==0 && origin1(i,j)==0 && origin2(i,j)==0  %黑  黑  黑
                    image1(son_x,son_y)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;
                
                elseif origin(i,j)==0 && origin1(i,j)==0 && origin2(i,j)~=0  %黑 黑 白
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;

                elseif origin(i,j)==0 && origin1(i,j)~=0 && origin2(i,j)==0  %黑 白 黑
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                
                elseif origin(i,j)==0 && origin1(i,j)~=0 && origin2(i,j)~=0  %黑 白 白
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)==0 && origin2(i,j)==0  %白 黑 黑
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;

                elseif origin(i,j)~=0 && origin1(i,j)==0 && origin2(i,j)~=0  %白 黑 白
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)~=0 && origin2(i,j)~=0  %白 白 白
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)~=0 && origin2(i,j)==0  %白 白 黑
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;
                
                end
            case 2
                if origin(i,j)==0 && origin1(i,j)==0 && origin2(i,j)==0  %黑  黑  黑
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;
                
                elseif origin(i,j)==0 && origin1(i,j)==0 && origin2(i,j)~=0  %黑 黑 白
                    image1(son_x,son_y)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)==0 && origin1(i,j)~=0 && origin2(i,j)==0  %黑 白 黑
                    image1(son_x,son_y)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;
                
                elseif origin(i,j)==0 && origin1(i,j)~=0 && origin2(i,j)~=0  %黑 白 白
                    image1(son_x,son_y)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)==0 && origin2(i,j)==0  %白 黑 黑
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)==0 && origin2(i,j)~=0  %白 黑 白
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)~=0 && origin2(i,j)~=0  %白 白 白
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)~=0 && origin2(i,j)==0  %白 白 黑
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;
                
                end
                
            case 3
                if origin(i,j)==0 && origin1(i,j)==0 && origin2(i,j)==0  %黑  黑  黑
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;
                
                elseif origin(i,j)==0 && origin1(i,j)==0 && origin2(i,j)~=0  %黑 黑 白
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)==0 && origin1(i,j)~=0 && origin2(i,j)==0  %黑 白 黑
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;
                
                elseif origin(i,j)==0 && origin1(i,j)~=0 && origin2(i,j)~=0  %黑 白 白
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;

                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)==0 && origin2(i,j)==0  %白 黑 黑
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)==0 && origin2(i,j)~=0  %白 黑 白
                    image1(son_x,son_y)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)~=0 && origin2(i,j)~=0  %白 白 白
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)~=0 && origin2(i,j)==0  %白 白 黑
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                
                end
                
            case 4
                if origin(i,j)==0 && origin1(i,j)==0 && origin2(i,j)==0  %黑  黑  黑
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                
                elseif origin(i,j)==0 && origin1(i,j)==0 && origin2(i,j)~=0  %黑 黑 白
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)==0 && origin1(i,j)~=0 && origin2(i,j)==0  %黑 白 黑
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;
                
                elseif origin(i,j)==0 && origin1(i,j)~=0 && origin2(i,j)~=0  %黑 白 白
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)==0 && origin2(i,j)==0  %白 黑 黑
                    image1(son_x,son_y)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)==0 && origin2(i,j)~=0  %白 黑 白
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)~=0 && origin2(i,j)~=0  %白 白 白
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;

                elseif origin(i,j)~=0 && origin1(i,j)~=0 && origin2(i,j)==0  %白 白 黑
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;
                
                end

        end
    end
end

end


function image = merge(image1,image2)
Size=size(image1);
x=Size(1);
y=Size(2);

% 创建一个大小为x by y的全零矩阵
image=zeros(x,y);
image(:,:)=255;

for i=1:x
    for j=1:y
        % 将两个子图像逐像素进行AND操作，得到还原后的图像
        image(i,j)=image1(i,j)&image2(i,j);
    end
end

end