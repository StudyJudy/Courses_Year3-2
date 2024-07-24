function ImageHiding()
    x=imread("boy.bmp"); %载体图像
    m=imread("snowman.bmp"); %水印图像
    imshow(x,[])
    % title("boy_origin.bmp")
    imshow(m,[]);
    % title("snowman_origin.bmp")
    WaterMarked=Embedding(x,m);% 嵌入水印
    watermark=Decode(WaterMarked);
end

function WaterMarked = Embedding(origin,watermark)
    [Mc,Nc]=size(origin);
    % 创建一个与载体图像大小一致的全0矩阵
    WaterMarked=uint8(zeros(size(origin)));
    
    for i=1:Mc
        for j=1:Nc
            % 将水印图像的每个像素点的最低有效位嵌入到载体图像中对应像素点的最低有效位
            WaterMarked(i,j)=bitset(origin(i,j),1,watermark(i,j));
        end
    end
    
    imwrite(WaterMarked,'lsb_embedded_image_watermark.bmp','bmp');
    figure;
    imshow(WaterMarked,[]);
    % title("The embedded watermarked image");
end

%提取伪装图像的最低位平面，恢复隐藏的图像 
function WaterMark=Decode(WaterMarked)
    [Mc,Nc]=size(WaterMarked);
     % 创建一个与嵌入水印后的图像大小一致的全0矩阵
    WaterMark=uint8(zeros(size(WaterMarked)));

    for i=1:Mc
        for j=1:Nc
            % 提取每个像素点的最低有效位，并存储到watermaek矩阵的对应位置中
            WaterMark(i,j)=bitget(WaterMarked(i,j),1);
        end
    end

    imwrite(WaterMark,'lsb_extracted_image_watermark.bmp','bmp');
    figure;
    imshow(WaterMark,[]);
    % title("The decoded watermark image")

end