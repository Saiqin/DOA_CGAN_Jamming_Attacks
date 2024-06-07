# import pdb
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio


def getargs():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # traffic config
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--block_size', type=int, default=10)
    parser.add_argument('--dir_name', type=str, default="test")
    opt = parser.parse_args()

    # np.set_printoptions(precision=2)
    return opt


def load_matdata(filename):
    mat_contents = sio.loadmat(file_name=filename)
    train_r = mat_contents['Train_R'].reshape([-1, 56])
    train_r_clean = mat_contents['Train_R_clean'].reshape([-1, 56])
    return (torch.FloatTensor(train_r_clean), torch.FloatTensor(train_r), None, None)


def load_matdata_test(filename):
    mat_contents = sio.loadmat(file_name=filename)
    test_r = mat_contents['Test_R'].reshape([-1, 56])
    test_r_clean = mat_contents['Test_R_clean'].reshape([-1, 56])

    return (torch.FloatTensor(test_r_clean), torch.FloatTensor(test_r), None, None)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, condition=None):
        residual = condition
        out = self.conv1(x)
        # residual = out1
        out = self.relu(out)
        out = self.conv2(out)
        print(out.shape, residual.shape)
        out += residual
        out = self.relu(out)
        return out


class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.conv1 = ResidualBlock(1, 32)
        # self.conv2 = ResidualBlock(32, 64)
        self.fc1 = nn.Linear(64*7*8, 128)
        self.fc2 = nn.Linear(128, output_size)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.tanh = nn.Tanh()
        # nn.Linear()

    def forward(self, x, condition=None):
        x = x.reshape([-1, 1, 7, 8])
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # x = self.conv1(x, x)
        # x = self.conv2(x, x)
        x = torch.flatten(x, 1)
        # print(x.shape, "1")
        x = torch.relu(self.fc1(x))
        # print(x.shape, "2")
        x = self.fc2(x)
        # print(x.shape, "3")
        # x = self.tanh(x)
        return x


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, classes):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.conv1 = ResidualBlock(1, 32)
        # self.conv2 = ResidualBlock(32, 64)
        # self.conv3 = ResidualBlock(64, 32)
        self.fc1 = nn.Linear(32*7*8, 128)
        self.fc2 = nn.Linear(128, classes)
        self.sigmod = nn.Sigmoid()

    def forward(self, x, condition=None):
        x = torch.stack([x.reshape([-1, 1, 7, 8]),
                        condition.reshape([-1, 1, 7, 8])], dim=1).squeeze(dim=2)
        # x = x.reshape([-1, 1, 7, 8])
        # condition = condition.reshape([-1, 1, 7, 8])
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # x = self.conv1(x, condition)
        # x = self.conv2(x, condition)
        # x = self.conv3(x, condition)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmod(x)
        return x


class TraninDataset(Dataset):
    def __init__(self, xn, xc):
        super().__init__()
        self.xn = xn
        self.xc = xc

    def __len__(self):
        return len(self.xn)

    def __getitem__(self, idx):
        return self.xn[idx], self.xc[idx]


def main():
    # 训练参数
    opt = getargs()
    # Check device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = opt.epoch_num
    block = opt.block_size
    batch_size = opt.batch_size
    # input_size = 128
    hidden_dim = 256
    data_size = 56  # (7 * 8)
    latent_size = data_size
    classes = 10
    output_size_d = 1
    test_image_num = 10

    # 初始化模型
    generator = Generator(latent_size, hidden_dim, data_size).to(device)
    discriminator = Discriminator(
        data_size, hidden_dim, output_size_d).to(device)

    # 损失函数和优化器
    criterion = nn.BCELoss()
    criterion_MSE = nn.MSELoss(reduction='sum')
    # criterion = nn.CrossEntropyLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    # output documents
    # output_path = "./outputs/res{}".format(opt.dir_name)
    output_path = "./outputs/res_condition_epoch{}_batchsize{}_blocksize{}_{}".format(
        epochs, batch_size, block, opt.dir_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    fake_samples = list()
    real_samples = list()

    fake_samples_test = list()
    real_samples_test = list()

    X_c, X_n, _, _ = load_matdata("./data/data_net.mat")
    X_c_test, X_n_test, _, _ = load_matdata_test("./data/data_net.mat")
    y_real = torch.ones([X_c.shape[0], 1]).to(device)
    y_fake = torch.zeros([X_c.shape[0], 1]).to(device)
    y_real_test = torch.ones([X_c_test.shape[0], 1]).to(device)
    y_fake_test = torch.zeros([X_c_test.shape[0], 1]).to(device)

    X_noisy, X_clean = X_n.to(device), X_c.to(device)
    X_noisy_test, X_clean_test = X_n_test.to(device), X_c_test.to(device)

    trainset = TraninDataset(X_n, X_c)
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True)
    loss_d_list = list()
    loss_g_list = list()
    real_score_list = list()
    fake_score_list = list()

    loss_d_list_test = list()
    loss_g_list_test = list()
    real_score_list_test = list()
    fake_score_list_test = list()

    rand_index_test = np.random.randint(0, X_c_test.shape[0], test_image_num)

    for epoch in range(epochs):
        fake_outputs = np.array([])
        # print(id,"----")
        generator.train()
        discriminator.train()
        # real_outputs = list()
        for id, (X_noisy, X_clean) in enumerate(train_loader):
            X_noisy, X_clean = X_noisy.to(device), X_clean.to(device)
            y_real = torch.ones((X_clean.shape[0], 1)).to(device)
            y_fake = torch.zeros((X_clean.shape[0], 1)).to(device)

            # 每个epoch都生成新的数据
            # print(X_clean.shape, X_noisy.shape, y_real.shape, y_fake.shape)

            # 训练判别器
            # discriminator.zero_grad()
            # 真实数据的损失
            pred_real = discriminator(X_clean, X_clean)
            # import pdb;pdb.set_trace()
            loss_real = criterion(pred_real, y_real)
            # real_score = pred_real.mean().item()
            real_score = pred_real

            # 假数据的损失
            # X_noisy = torch.randn([batch_size, latent_size]).to(device)
            X_fake = generator(X_noisy)
            # y_fake = torch.zeros(batch_size, 1)
            # import pdb;pdb.set_trace()
            pred_fake = discriminator(X_fake, X_clean)
            loss_fake = criterion(pred_fake, y_fake)
            fake_score = pred_fake
            loss_label = 0  # criterion_MSE(X_fake,X_clean)

            # 判别器的总损失
            loss_d = (loss_real + loss_fake)/2 + loss_label
            # d_loss_train += loss_d
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # 训练生成器
            # generator.zero_grad()
            # X_noisy = torch.randn([batch_size, latent_size]).to(device)
            fake_sample = generator(X_noisy)
            # fake_outputs.append(fake_sample.cpu().detach().numpy())
            fake_outputs = np.append(
                fake_outputs, fake_sample.cpu().detach().numpy())
            pred_fake = discriminator(fake_sample, X_clean)
            loss_g = criterion(pred_fake, y_real)  # 生成器希望判别器将其生成的假数据视为真实数据
            # g_loss_train += loss_g
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            if (id+1) % 100 == 0 or id == 0:
                print("Epoch : {:05d}, Batch : {:05d}, loss_d : {:.4f}, loss_g : {:.4f}, real_score : {:.4f}, fake_score : {:.4f}".format(
                    epoch+1, id+1, loss_d.item(), loss_g.item(), real_score.mean().item(), fake_score.mean().item()))

        if (epoch+1) % block == 0 or epoch == 0:
            loss_d_list.append(loss_d.item())
            loss_g_list.append(loss_g.item())
            real_score_list.append(real_score.mean().item())
            fake_score_list.append(fake_score.mean().item())
            fake_samples.append(fake_outputs.reshape([X_c.shape[0], 56]))
            real_samples.append(X_c.cpu().detach().numpy())
            generator.eval()
            discriminator.eval()
            with torch.no_grad():
                # pass
                preal = discriminator(X_clean_test, X_clean_test)
                lreal = criterion(preal, y_real_test)
                rscore = preal

                xfake = generator(X_noisy_test)
                fsample = xfake
                pfake = discriminator(xfake, X_clean_test)
                lfake = criterion(pfake, y_fake_test)
                fscore = pfake

                ldtest = (lreal+lfake)/2.
                lgtest = criterion(pfake, y_real_test)

                fake_samples_test.append(fsample.cpu().detach().numpy())
                real_samples_test.append(X_c_test.cpu().detach().numpy())
                print("Epoch : {:05d}, loss_d : {:.4f}, loss_g : {:.4f}, real_score : {:.4f}, fake_score : {:.4f}".format(
                    epoch+1, ldtest.item(), lgtest.item(), rscore.mean().item(), fscore.mean().item()))
                loss_d_list_test.append(ldtest.item())
                loss_g_list_test.append(lgtest.item())
                real_score_list_test.append(rscore.mean().item())
                fake_score_list_test.append(fscore.mean().item())

                for idx in rand_index_test:
                    test_image_path = os.path.join(
                        output_path, "test{}".format(idx))
                    if not os.path.exists(test_image_path):
                        os.mkdir(test_image_path)
                    plt.plot(real_samples_test[-1][idx], label="clean")
                    plt.plot(fake_samples_test[-1][idx], label="gen_data")
                    plt.plot(X_n_test.cpu().detach().numpy()
                             [idx], label="noisy")
                    plt.legend()
                    plt.savefig(os.path.join(
                        test_image_path, "testdatafig{:05d}.png".format(epoch)))
                    # plt.show()
                    plt.close()

    real_samples_list = list()
    fake_samples_list = list()
    nois_samples_list = list()
    for i in range(len(fake_samples)):
        idx = np.random.randint(0, fake_samples[i].shape[0])
        real_samples_list.append(real_samples[i][idx].tolist())
        fake_samples_list.append(fake_samples[i][idx].tolist())
        nois_samples_list.append(X_n.cpu().detach().numpy()[idx].tolist())
        plt.plot(real_samples[i][idx], label="clean")
        plt.plot(fake_samples[i][idx], label="gen_data")
        plt.plot(X_n.cpu().detach().numpy()[idx], label="noisy")
        plt.legend()
        plt.savefig(os.path.join(
            output_path, "traindatafig{}.png".format(i*block)))
        # plt.show()
        plt.close()
    real_samples_list = np.array(real_samples_list)
    fake_samples_list = np.array(fake_samples_list)
    nois_samples_list = np.array(nois_samples_list)
    loss_d_list = np.array(loss_d_list)
    loss_g_list = np.array(loss_g_list)
    real_score_list = np.array(real_score_list)
    fake_score_list = np.array(fake_score_list)

    sio.savemat(os.path.join(output_path, "train.mat"), {"real_samples_list": real_samples_list, "fake_samples_list": fake_samples_list,
                "nois_samples_list": nois_samples_list, "loss_d_list": loss_d_list, "loss_g_list": loss_g_list, "real_score_list": real_score_list, "fake_score_list": fake_score_list})

    test_mat_dict = dict()

    for id in range(len(rand_index_test)):
        index = rand_index_test[id]
        real_samples_list_test = list()
        fake_samples_list_test = list()
        nois_samples_list_test = list()
        for i in range(len(fake_samples_test)):
            # idx = np.random.randint(0, fake_samples_test[i].shape[0])
            # str1 = str(real_samples_test[i][idx].tolist())
            # str2 = str(fake_samples_test[i][idx].tolist())
            real_samples_list_test.append(
                real_samples_test[i][id].tolist())
            fake_samples_list_test.append(
                fake_samples_test[i][id].tolist())
            nois_samples_list_test.append(
                X_n_test.cpu().detach().numpy()[id].tolist())

            # plt.plot(real_samples_test[i][idx], label="clean")
            # plt.plot(fake_samples_test[i][idx], label="gen_data")
            # plt.plot(X_n_test.cpu().detach().numpy()[idx], label="noisy")
            # plt.legend()
            # plt.savefig(os.path.join(
            #     output_path, "testdatafig{}.png".format(i*block)))
            # # plt.show()
            # plt.close()
        test_mat_dict["real_samples_list_test_{:06d}".format(
            index)] = np.array(real_samples_list_test)
        test_mat_dict["fake_samples_list_test_{:06d}".format(
            index)] = np.array(fake_samples_list_test)
        # print(np.array(fake_samples_list_test).shape)
        test_mat_dict["nois_samples_list_test_{:06d}".format(
            index)] = np.array(nois_samples_list_test)

    test_mat_dict["real_samples_list_test"] = real_samples_test[-1]
    test_mat_dict["fake_samples_list_test"] = fake_samples_test[-1]
    test_mat_dict["nois_samples_list_test"] = X_n_test.cpu(
    ).detach().numpy().tolist()
    # real_samples_list_test = np.array(real_samples_list_test)
    # fake_samples_list_test = np.array(fake_samples_list_test)
    # nois_samples_list_test = np.array(nois_samples_list_test)
    # loss_d_list_test = np.array(loss_d_list_test)
    # loss_g_list_test = np.array(loss_g_list_test)
    # real_score_list_test = np.array(real_score_list_test)
    # fake_score_list_test = np.array(fake_score_list_test)
    # print(fake_samples_list_test.shape)
    # sio.savemat(os.path.join(output_path, "test.mat"), {"real_samples_list_test": real_samples_list_test, "fake_samples_list_test": fake_samples_list_test,
    #                                                     "nois_samples_list_test": nois_samples_list_test, "loss_d_list_test": loss_d_list_test,
    #                                                     "loss_g_list_test": loss_g_list_test, "real_score_list_test": real_score_list_test,
    #                                                     "fake_score_list_test": fake_score_list_test, "rand_index_test": rand_index_test})
    sio.savemat(os.path.join(output_path, "test.mat"), test_mat_dict)

    # Save the model checkpoints
    torch.save(generator.state_dict(), os.path.join(output_path, 'G.ckpt'))
    torch.save(discriminator.state_dict(), os.path.join(output_path, 'D.ckpt'))


if __name__ == "__main__":
    main()
