%%%%===================DBF算法==================%%%%
%%%%==搜索范围、回波、阵元位置、波数==%%%%
function [fai_DBF,spe_out]=DBF(fai_range,X,M)
%导向矢量，第一维不同的阵元，第二维不同的搜索角度
if size(X,1)~=size(X,2)
    X=X*X';
end
A_DBF=exp(1j*pi*[0:M-1]'*(sind(fai_range)));
Yt_DBF=abs(sum(A_DBF'*X*A_DBF,2));
fai_DBF=fai_range(find(Yt_DBF==max(Yt_DBF),1));
spe_out=Yt_DBF/max(Yt_DBF);
% figure;
% plot(fai_DBF_ss,spe_out);
end
