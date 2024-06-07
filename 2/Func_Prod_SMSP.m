function jamming=Func_Prod_SMSP(tau,B,fs,k)
%%--------------产生SMSP干扰信号的程序-------------------
% B              雷达信号带宽
% T              雷达信号时宽
% fs              采样频率
% k              子波形复制的个数

T1=tau/k;
N1=round(fs*T1);
t1=(0:N1-1)/fs;
mu1=B/T1;
lfm1=exp(1i*pi*mu1*t1.^2);
jamming=repmat(lfm1,1,k);   %干扰信号
