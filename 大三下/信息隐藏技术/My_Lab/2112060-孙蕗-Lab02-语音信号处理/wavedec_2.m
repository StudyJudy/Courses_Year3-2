[a,fs]=audioread ("testaudio.wav") ;
[ca1,cd1]=wavedec(a(:,1),1 ,'db4') ;
a0=waverec(ca1,cd1,'db4') ;
%绘图
subplot (2 ,2 ,1) ; plot ( a ( : , 1)) ; 
subplot (2 ,2 ,2) ; plot ( cd1 ) ; %细节分量
subplot (2 ,2 ,3) ; plot ( ca1 ) ; %近似分量
subplot (2 ,2 ,4) ; plot ( a0 ) ; 
axes_handle = get ( gcf,'children') ;
axes ( axes_handle (4) ) ; title('Raw wave') ;
axes ( axes_handle (3) ) ; title('Detail Component') ;
axes ( axes_handle (2) ) ; title('Approximate Component') ;
axes ( axes_handle (1) ) ; title('Recovered wave') ;