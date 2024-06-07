function [DOAs,spe]=MUSIC_DOA(angle_range,X,M,source_num)
if size(X,1)~=size(X,2)
    R=1/size(X,2)*X*X';
else
    R=X;
end
[V,D]=eig(R);
diag_D=diag(abs(D));
[B,I]=sort(diag_D);
un=V(:,I(1:(end-source_num)));
spe=zeros(1,length(angle_range));
for ii=1:length(angle_range)
    a_sita=exp(1j*pi*[0:M-1]'*sind(angle_range(ii)));
    spe(ii)=1/abs(a_sita'*un*un'*a_sita);
end
spe=spe/max(spe);
%place=find(spe==max(spe),1);
[pks,locs] = findpeaks(spe);
[Value,In] = sort(pks,'descend');

DOAs = sort(angle_range(locs(In(1:source_num))));
end