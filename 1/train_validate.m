clear all;clc;close all;
M=8;
load test.mat
load Yr.mat DOA_target_test Test_Yt
fai_range = [3:.01:5.5];
for i = 1:length(DOA_target_test)
    Rx = real_imag_consits(fake_samples_list_test(i,:),ones(1,M),M);
   %Rx = real_imag_consits(Test_R_clean(i,:),ones(1,M),M);
    [DOA_DBF(i,1),spe] = DBF(fai_range,Rx,M); 
%     figure;hold on;
%     plot(fake_samples_list_test(i,:),'g-');
%     plot(real_samples_list_test(i,:),'r-');
%     close all;
end
RMSE = sqrt(mean((DOA_DBF-DOA_target_test).^2))

num = int32([551:1:704]);
figure;grid on;hold on;box on;
plot(1:length(num),DOA_test(num,:),'.','linewidth',4);
plot(1:length(num),DOA_DBF(num,:),'o','color',"#77AC30");
ylim([-60 60])
legend({'Jammer','Target','Train'},'FontSize',12,'Fontname','Times New Roman','FontWeight','bold')
xlabel({'Sample index'},'FontSize',12,'Fontname','Times New Roman','FontWeight','bold')
ylabel({'\theta(\circ)'},'FontSize',12,'Fontname','Times New Roman','FontWeight','bold');
%% sample
num = 480;
Rx = real_imag_consits(test_re_im(num,:),ones(1,M),M);
Yr = Test_Yt{num};
[~,spe_dnn] = DBF(fai_range,Rx,M);    
[~,spe_jam] = DBF(fai_range,Yr,M);  
[~,spe_dnn2] = MUSIC_DOA(fai_range,Rx,M,2);
[~,spe_jam2] = MUSIC_DOA(fai_range,Yr,M,2); 
figure;grid on;hold on;box on;
plot(fai_range,spe_jam,'b:','linewidth',1.5);
plot(fai_range,spe_dnn,'g-','linewidth',1.5);
plot([DOA_test(num,1)],[1],'ko','linewidth',1.5);
plot([DOA_test(num,2)],[1],'ro','linewidth',1.5);
%plot([DOA_test(num,1) DOA_test(num,1)],[0 1],'r-','linewidth',1.5);
%plot([DOA_test(num,2) DOA_test(num,2)],[0 1],'g-','linewidth',1.5);
legend({'Origin signal','Train signal','Jammer','Target'},'FontSize',12,'Fontname','Times New Roman','FontWeight','bold')
xlabel({'\theta(\circ)'},'FontSize',16,'Fontname','Times New Roman','FontWeight','bold')
ylabel({'Normalized Spectrum'},'FontSize',16,'Fontname','Times New Roman','FontWeight','bold');

%% Phase
A_sita1=exp(1j*pi*[0:M-1]'*sind(DOA_test(num,1)));
A_sita2=exp(1j*pi*[0:M-1]'*sind(DOA_test(num,2)));
figure;grid on;hold on;box on;
plot(phase(A_sita1),'-d','linewidth',1.5);
plot(phase(A_sita2),'-o','linewidth',1.5);
plot(phase(Yr(:,1)),'-s','linewidth',1.5);
plot(phase(Rx(:,1)),'-x','linewidth',1.5);
legend({'Jammer','Target','Origin signal','Train signal'},'FontSize',12,'Fontname','Times New Roman','FontWeight','bold')
xlabel({'Array element'},'FontSize',16,'Fontname','Times New Roman','FontWeight','bold')
ylabel({'Rad(\pi)'},'FontSize',16,'Fontname','Times New Roman','FontWeight','bold');


figure;
subplot(4,1,1)
imagesc(feature_extract_R(A_sita1));
title('(a) Jammer','FontSize',12,'Fontname','Times New Roman','FontWeight','bold');
subplot(4,1,2)
imagesc(feature_extract_R(A_sita2));
title('(b) Target','FontSize',12,'Fontname','Times New Roman','FontWeight','bold');
subplot(4,1,3)
imagesc(feature_extract_R(A_sita1+A_sita2));
title('(c) Jammer + Target','FontSize',12,'Fontname','Times New Roman','FontWeight','bold');
subplot(4,1,4)
imagesc(test_re_im(num,:));
title('(d) Train','FontSize',12,'Fontname','Times New Roman','FontWeight','bold');

xlabel('Rx','position',[28,1.8,0],'FontSize',12,'Fontname','Times New Roman','FontWeight','bold');
ylabel('Magnitude','position',[-2.2,-1.6,0],'FontSize',12,'Fontname','Times New Roman','FontWeight','bold');







