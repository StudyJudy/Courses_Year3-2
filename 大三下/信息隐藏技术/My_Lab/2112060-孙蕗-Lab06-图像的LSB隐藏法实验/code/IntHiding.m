function IntHiding()
    x=imread("boy.bmp"); %载体图像
    m=2112060
    imshow(x,[])
    WaterMarked=Embedding(x,m);
    message=Decode(WaterMarked)
end

function WaterMarked = Embedding(origin,watermark)
    [Mc,Nc]=size(origin);
    % 创建一个与载体图像大小一致的全0矩阵
    WaterMarked=uint8(zeros(size(origin)));
    
    for i=1:Mc
        for j=1:Nc
            if i==1 && j<=22
                 % 将嵌入整数对应二进制位嵌入到载体图像中对应像素点的最低有效位
                bit=bitget(watermark,j);
                WaterMarked(i,j)=bitset(origin(i,j),1,bit);
            else
                WaterMarked(i,j)=origin(i,j);
            end
        end
    end
    
    imwrite(WaterMarked,'lsb_embedded_int_watermarked.bmp','bmp');
    figure;
    imshow(WaterMarked,[]);
    title("The embedded watermarked image");
end

function message=Decode(WaterMarked)
    message=0;
    for j=1:22
        bit=bitget(WaterMarked(1,j),1);
        message=bitset(message,j,bit);
    end
end