clc;close all;clear all;
M=8;
SNR = 10;
JSR = 15;
Tp=5e-6; %脉冲
B=5e6;  %带宽
Fs=B*2;  %采样率
%% LFM
LFM=Func_Prod_LFM(Tp,B,Fs);
%% SMSP
k = 5; % 子波形复制的个数
SMSP = Func_Prod_SMSP(Tp,B,Fs,k);
Tplen = round(Fs*Tp);
t = (0:Tplen-1)/Fs;

num  = 5000;
scope = -30:5:30;
Jam_DOA = repmat(scope,num,1) + [rand(num,1)*1-0.5];
DOA_AA  =7;
DOA_target = DOA_AA + [rand(num,1)*1-0.5];
DOA_target = repmat(DOA_target,length(scope),1);
Jam_DOA = Jam_DOA(:);
Train_Yt={};
for ii = 1:size(DOA_target,1)
    if mod(ii,1000)==0
        ii
    end
    A_sita1=exp(1j*pi*[0:M-1]'*sind(Jam_DOA(ii,1)));
    A_sita2=exp(1j*pi*[0:M-1]'*sind(DOA_target(ii,1)));
    noise1 = (randn(M,Tplen)+1j*randn(M,Tplen))*0.707;
    noise2 = (randn(M,Tplen)+1j*randn(M,Tplen))*0.707;
    X1=10^(SNR/20)*A_sita1* SMSP+ noise1;
    X2=10^(JSR/20)*A_sita2* LFM+ noise2;
    X = X1 + X2;
    [Train_R(ii,:),~]=feature_extract_R(X,1) ;
    Train_R_clean(ii,:) = feature_extract_R(A_sita2,0);
    Train_Yt = [Train_Yt;X];
end

%% test
num = 500;
Jam_DOA_test = repmat(scope,num,1) + [rand(num,1)*1-0.5];
DOA_target_test = DOA_AA + [rand(num,1)*1-0.5];
DOA_target_test = repmat(DOA_target_test,length(scope),1);
Jam_DOA_test = Jam_DOA_test(:);
Test_Yt={};
for ii=1:size(DOA_target_test,1)
    if mod(ii,1000)==0
        ii
    end
    A_sita1=exp(1j*pi*[0:M-1]'*sind(Jam_DOA_test(ii,1)));
    A_sita2=exp(1j*pi*[0:M-1]'*sind(DOA_target_test(ii,1)));
    noise1 = (randn(M,Tplen)+1j*randn(M,Tplen))*0.707;
    noise2 = (randn(M,Tplen)+1j*randn(M,Tplen))*0.707;
    X1=10^(SNR/20)*A_sita1* SMSP+ noise1;
    X2=10^(JSR/20)*A_sita2* LFM+ noise2;
    X = X1 + X2;
    [Test_R(ii,:),~]=feature_extract_R(X,1) ;
    Test_R_clean(ii,:) = feature_extract_R(A_sita2,0);
    Test_Yt = [Test_Yt;X];
end


%% Val
Jam_DOA_Val = Jam_DOA_test;
DOA_target_Val = DOA_target_test;
Val_Yt = {};
for ii=1:size(DOA_target_test,1)
    if mod(ii,1000)==0
        ii
    end
    A_sita1=exp(1j*pi*[0:M-1]'*sind(Jam_DOA_Val(ii,1)));
    A_sita2=exp(1j*pi*[0:M-1]'*sind(DOA_target_Val(ii,1)));
    noise1 = (randn(M,Tplen)+1j*randn(M,Tplen))*0.707;
    noise2 = (randn(M,Tplen)+1j*randn(M,Tplen))*0.707;
    X1=10^(SNR/20)*A_sita1* SMSP+ noise1;
    X2=10^(JSR/20)*A_sita2* LFM+ noise2;
    X = X1 + X2;
    [Val_R(ii,:),~]=feature_extract_R(X,1) ;
    Val_R_clean(ii,:) = feature_extract_R(A_sita2,0);
    Val_Yt = [Val_Yt;X];
end

save data_net.mat Train_R Train_R_clean Test_R Test_R_clean Val_R Val_R_clean
save Yr.mat Train_Yt Test_Yt Val_Yt Jam_DOA DOA_target DOA_target_test DOA_target_Val

