clear all;clc;close all;
M=8;
load test.mat
load Yr.mat DOA_target_test
num = 6500;
DOA1 = fake_samples_list_test(1:num,:);
DOA2 = fake_samples_list_test(num*1+1:num*2,:);
DOA3 = fake_samples_list_test(num*2+1:num*3,:);
DOA4 = fake_samples_list_test(num*3+1:num*4,:);
DOA5 = fake_samples_list_test(num*4+1:num*5,:);
DOA6 = fake_samples_list_test(num*5+1:num*6,:);
DOA7 = fake_samples_list_test(num*6+1:num*7,:);
DOA8 = fake_samples_list_test(num*7+1:num*8,:);

D1 = DOA_target_test(1:num);
D2 = DOA_target_test(num*1+1:num*2);
D3 = DOA_target_test(num*2+1:num*3);
D4 = DOA_target_test(num*3+1:num*4);
D5 = DOA_target_test(num*4+1:num*5);
D6 = DOA_target_test(num*5+1:num*6);
D7 = DOA_target_test(num*6+1:num*7);
D8 = DOA_target_test(num*7+1:num*8);

DOA_sample = [-28 -22 -17 -3 7 13 18 23];
fai_range = [DOA_sample(1)-1.5:0.01:DOA_sample(1)+1.5];
%% note that can also use MUSIC
for i = 1:num
    Rx = real_imag_consits(DOA1(i,:),ones(1,M),M);
    [DOA_DBF1(i,1),spe] = MUSIC_DOA(fai_range,Rx,M,1);
end
fai_range = [DOA_sample(2)-1.5:0.01:DOA_sample(2)+1.5];
for i = 1:num
    Rx = real_imag_consits(DOA2(i,:),ones(1,M),M);
    [DOA_DBF2(i,1),spe] = MUSIC_DOA(fai_range,Rx,M,1);
end
fai_range = [DOA_sample(3)-1.5:0.01:DOA_sample(3)+1.5];
for i = 1:num
    Rx = real_imag_consits(DOA3(i,:),ones(1,M),M);
    [DOA_DBF3(i,1),spe] = MUSIC_DOA(fai_range,Rx,M,1);
end
fai_range = [DOA_sample(4)-1.5:0.01:DOA_sample(4)+1.5];
for i = 1:num
    Rx = real_imag_consits(DOA4(i,:),ones(1,M),M);
    [DOA_DBF4(i,1),spe] = MUSIC_DOA(fai_range,Rx,M,1);
end
fai_range = [DOA_sample(5)-1.5:0.01:DOA_sample(5)+1.5];
for i = 1:num
    Rx = real_imag_consits(DOA5(i,:),ones(1,M),M);
    [DOA_DBF5(i,1),spe] = MUSIC_DOA(fai_range,Rx,M,1);
end
fai_range = [DOA_sample(6)-1.5:0.01:DOA_sample(6)+1.5];
for i = 1:num
    Rx = real_imag_consits(DOA6(i,:),ones(1,M),M);
    [DOA_DBF6(i,1),spe] = MUSIC_DOA(fai_range,Rx,M,1);
end
fai_range = [DOA_sample(7)-1.5:0.01:DOA_sample(7)+1.5];
for i = 1:num
    Rx = real_imag_consits(DOA7(i,:),ones(1,M),M);
    [DOA_DBF7(i,1),spe] = MUSIC_DOA(fai_range,Rx,M,1);
end
fai_range = [DOA_sample(8)-1.5:0.01:DOA_sample(8)+1.5];
for i = 1:num
    Rx = real_imag_consits(DOA8(i,:),ones(1,M),M);
    [DOA_DBF8(i,1),spe] = MUSIC_DOA(fai_range,Rx,M,1);
end


RMSE1 = sqrt(mean((DOA_DBF1-D1).^2))
RMSE2 = sqrt(mean((DOA_DBF2-D2).^2))
RMSE3 = sqrt(mean((DOA_DBF3-D3).^2))
RMSE4 = sqrt(mean((DOA_DBF4-D4).^2))
RMSE5 = sqrt(mean((DOA_DBF5-D5).^2))
RMSE6 = sqrt(mean((DOA_DBF6-D6).^2))
RMSE7 = sqrt(mean((DOA_DBF7-D7).^2))
RMSE8 = sqrt(mean((DOA_DBF8-D8).^2))

save cGan.mat RMSE1 RMSE2 RMSE3 RMSE4 RMSE5 RMSE6 RMSE7 RMSE8



