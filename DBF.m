%%%%===================DBF�㷨==================%%%%
%%%%==������Χ���ز�����Ԫλ�á�����==%%%%
function [fai_DBF,spe_out]=DBF(fai_range,X,M)
%����ʸ������һά��ͬ����Ԫ���ڶ�ά��ͬ�������Ƕ�
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
