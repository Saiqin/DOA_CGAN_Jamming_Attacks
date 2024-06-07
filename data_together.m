clc;clear all;
load .\1\data_net.mat
AA1 = Train_R; AA2=Train_R_clean;;
BB1 = Test_R; BB2=Test_R_clean;;
CC1=Val_R;CC2=Val_R_clean;
load .\2\data_net.mat
AA1 = [AA1;Train_R];AA2=[AA2;Train_R_clean];%AA3 = [AA3;Train_label];
BB1 = [BB1;Test_R];BB2=[BB2;Test_R_clean];%BB3 = [BB3;Test_label];
CC1 = [CC1;Val_R];CC2=[CC2;Val_R_clean];%CC3 = [CC3;Val_label];
load .\3\data_net.mat
AA1 = [AA1;Train_R];AA2=[AA2;Train_R_clean];%AA3 = [AA3;Train_label];
BB1 = [BB1;Test_R];BB2=[BB2;Test_R_clean];%BB3 = [BB3;Test_label];
CC1 = [CC1;Val_R];CC2=[CC2;Val_R_clean];%CC3 = [CC3;Val_label];
load .\4\data_net.mat
AA1 = [AA1;Train_R];AA2=[AA2;Train_R_clean];%AA3 = [AA3;Train_label];
BB1 = [BB1;Test_R];BB2=[BB2;Test_R_clean];%BB3 = [BB3;Test_label];
CC1 = [CC1;Val_R];CC2=[CC2;Val_R_clean];%CC3 = [CC3;Val_label];
load .\5\data_net.mat
AA1 = [AA1;Train_R];AA2=[AA2;Train_R_clean];%AA3 = [AA3;Train_label];
BB1 = [BB1;Test_R];BB2=[BB2;Test_R_clean];%BB3 = [BB3;Test_label];
CC1 = [CC1;Val_R];CC2=[CC2;Val_R_clean];%CC3 = [CC3;Val_label];
load .\6\data_net.mat
AA1 = [AA1;Train_R];AA2=[AA2;Train_R_clean];%AA3 = [AA3;Train_label];
BB1 = [BB1;Test_R];BB2=[BB2;Test_R_clean];%BB3 = [BB3;Test_label];
CC1 = [CC1;Val_R];CC2=[CC2;Val_R_clean];%CC3 = [CC3;Val_label];
load .\7\data_net.mat
AA1 = [AA1;Train_R];AA2=[AA2;Train_R_clean];%AA3 = [AA3;Train_label];
BB1 = [BB1;Test_R];BB2=[BB2;Test_R_clean];%BB3 = [BB3;Test_label];
CC1 = [CC1;Val_R];CC2=[CC2;Val_R_clean];%CC3 = [CC3;Val_label];
load .\8\data_net.mat
AA1 = [AA1;Train_R];AA2=[AA2;Train_R_clean];%AA3 = [AA3;Train_label];
BB1 = [BB1;Test_R];BB2=[BB2;Test_R_clean];%BB3 = [BB3;Test_label];
CC1 = [CC1;Val_R];CC2=[CC2;Val_R_clean];%CC3 = [CC3;Val_label];

Train_R=AA1;Train_R_clean=AA2;%Train_label=AA3;
Test_R=BB1;Test_R_clean=BB2;%Test_label=BB3;
Val_R=CC1;Val_R_clean=CC2;%Val_label=CC3;
save data_net.mat Train_R Train_R_clean Test_R Test_R_clean Val_R Val_R_clean

load .\1\Yr.mat DOA_target_test
AA = DOA_target_test;
load .\2\Yr.mat DOA_target_test
AA = [AA;DOA_target_test];
load .\3\Yr.mat DOA_target_test
AA = [AA;DOA_target_test];
load .\4\Yr.mat DOA_target_test
AA = [AA;DOA_target_test];
load .\5\Yr.mat DOA_target_test
AA = [AA;DOA_target_test];
load .\6\Yr.mat DOA_target_test
AA = [AA;DOA_target_test];
load .\7\Yr.mat DOA_target_test
AA = [AA;DOA_target_test];
load .\8\Yr.mat DOA_target_test
AA = [AA;DOA_target_test];

DOA_target_test = AA;
save Yr.mat DOA_target_test





