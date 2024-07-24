function Superposition_Binary01()
path = input("请输入二值图像叠相术要加密的图片：",'s');% boy.jpg
origin=imread(path);
% 二值图像
origin_bw = im2bw(origin);
figure('name','boy二值图像');
imshow(origin_bw);
imwrite(origin_bw,'boy_binary_image.jpg');

path = input("请输入二值图像叠相术隐藏图1：",'s');%flower.jpg
origin1=imread(path);
% 二值图像
origin1_bw = im2bw(origin1);
figure('name','flower二值图像叠相术隐藏图1');
imshow(origin1_bw);
imwrite(origin1_bw,'flower_hiding_binary_image.jpg');

path = input("请输入二值图像叠相术隐藏图2：",'s');%snowman.jpg
origin2 = imread(path);
% 二值图像
origin2_bw = im2bw(origin2);
figure('name','flower二值图像叠相术隐藏图2');
imshow(origin2_bw);
imwrite(origin2_bw,'snowman_hiding_binary_image.jpg');

[image1,image2] = decompose(origin_bw,origin1_bw,origin2_bw);
figure('name','二值图像叠相术分存图1');
imshow(image1);
imwrite(image1,'binary_camouflage1.jpg');

figure('name','二值图像叠相术分存图2');
imshow(image2);
imwrite(image2,'binary_camouflage2.jpg');

figure('name','二值图像叠相术恢复图');
originB = merge(image1,image2);
imshow(originB);
imwrite(originB,'binary_restored.jpg');

% 缩放
% originB_downsize=imresize(originB,0.5);
% imshow(originB_downsize);

end

% decompose函数将二值图像分解成两个子图像
function [image1,image2] = decompose(origin_bw,origin1_bw,origin2_bw)
%初始化
Size=size(origin_bw);
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
        % 产生一个1-4之间的随机数，任选分存方案
        key = randi(4);

        % 计算子图的起始横坐标
        % 根据当前像素在原始图像中的位置，计算出其在子图中的位置
        son_x=1+2*(i-1);

        % 计算子图的起始纵坐标
        % 根据当前像素在原始图像中的位置，计算出其在子图中的位置
        son_y=1+2*(j-1);
        switch key
            case 1
                 %黑  黑  黑
                if origin_bw(i,j)==0 && origin1_bw(i,j)==0 && origin2_bw(i,j)==0 
                    image1(son_x,son_y)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;
                
                %黑 黑 白
                elseif origin_bw(i,j)==0 && origin1_bw(i,j)==0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;

                %黑 白 黑
                elseif origin_bw(i,j)==0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)==0  
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                
                %黑 白 白
                elseif origin_bw(i,j)==0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)~=0  
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;

                %白 黑 黑
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)==0 && origin2_bw(i,j)==0  
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;

                %白 黑 白
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)==0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;

                %白 白 白
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;

                %白 白 黑
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)==0  
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;
                
                end
            case 2
                 %黑  黑  黑
                if origin_bw(i,j)==0 && origin1_bw(i,j)==0 && origin2_bw(i,j)==0 
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;
                
                %黑 黑 白
                elseif origin_bw(i,j)==0 && origin1_bw(i,j)==0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;

                %黑 白 黑
                elseif origin_bw(i,j)==0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)==0 
                    image1(son_x,son_y)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;
                
                %黑 白 白
                elseif origin_bw(i,j)==0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;

                %白 黑 黑
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)==0 && origin2_bw(i,j)==0  
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;

                %白 黑 白
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)==0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                %白 白 白
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                %白 白 黑
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)==0  
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;
                
                end
                
            case 3
                %黑 黑 黑
                if origin_bw(i,j)==0 && origin1_bw(i,j)==0 && origin2_bw(i,j)==0  
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;
                
                %黑 黑 白
                elseif origin_bw(i,j)==0 && origin1_bw(i,j)==0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                %黑 白 黑
                elseif origin_bw(i,j)==0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)==0  
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;
                
                %黑 白 白
                elseif origin_bw(i,j)==0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;

                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                %白 黑 黑
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)==0 && origin2_bw(i,j)==0  
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                %白 黑 白
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)==0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                %白 白 白
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                %白 白 黑
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)==0  
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                
                end
                
            case 4
                %黑 黑 黑
                if origin_bw(i,j)==0 && origin1_bw(i,j)==0 && origin2_bw(i,j)==0  
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y)=0;
                
                %黑 黑 白
                elseif origin_bw(i,j)==0 && origin1_bw(i,j)==0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y)=0;
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                %黑 白 黑
                elseif origin_bw(i,j)==0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)==0  
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;
                
                %黑 白 白
                elseif origin_bw(i,j)==0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                %白 黑 黑
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)==0 && origin2_bw(i,j)==0  
                    image1(son_x,son_y)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y)=0;
                    image2(son_x+1,son_y)=0;
                    image2(son_x+1,son_y+1)=0;

                %白 黑 白
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)==0 && origin2_bw(i,j)~=0  
                    image1(son_x,son_y+1)=0;
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;

                %白 白 白
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)~=0  
                    image1(son_x+1,son_y)=0;
                    image1(son_x+1,son_y+1)=0;

                    image2(son_x,son_y+1)=0;
                    image2(son_x+1,son_y+1)=0;

                %白 白 黑
                elseif origin_bw(i,j)~=0 && origin1_bw(i,j)~=0 && origin2_bw(i,j)==0  
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