from TGenerator import LSTMGenerator
from TDiscriminator import LSTMDiscriminator
import torch

if __name__ == '__main__':
    batch_size = 16
    seq_len = 32
    noise_dim = 100
    seq_dim = 4

    gen = LSTMGenerator(noise_dim, seq_dim)
    dis = LSTMDiscriminator(seq_dim)
    noise = torch.randn(8, 16, noise_dim)
    gen_out = gen(noise)
    dis_out = dis(gen_out)

    print("Noise: ", noise.size())
    print("Generator output: ", gen_out.size())
    print("Discriminator output: ", dis_out.size())
