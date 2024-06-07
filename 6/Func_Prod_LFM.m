function s=Func_Prod_LFM(tau,B,fs)
%     t = -tau/2:1/fs:tau/2-1/fs;
    t=0:1/fs:tau-1/fs;
    K = B/tau;
    s=exp(1j*pi*K*t.^2);
end