function jamming=Func_Prod_SMSP(tau,B,fs,k)
%%--------------����SMSP�����źŵĳ���-------------------
% B              �״��źŴ���
% T              �״��ź�ʱ��
% fs              ����Ƶ��
% k              �Ӳ��θ��Ƶĸ���

T1=tau/k;
N1=round(fs*T1);
t1=(0:N1-1)/fs;
mu1=B/T1;
lfm1=exp(1i*pi*mu1*t1.^2);
jamming=repmat(lfm1,1,k);   %�����ź�
