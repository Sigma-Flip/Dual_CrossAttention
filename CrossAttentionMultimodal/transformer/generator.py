import torch

if __name__ == "__main__":
    batch = 32
    seq_len = 3
    d_word =20
    n_heads = 4
    d_head = d_word // n_heads # 5
    data = torch.randn(batch, seq_len, d_word)
    data = data.view(batch, seq_len,n_heads, d_head )
    print(data.shape)
    d_t = torch.transpose(data, -2,-1)
    print(d_t.shape)
    w = 3
    h = 2
    data = torch.randn(w,h)
    prob = torch.softmax(data, dim = -1)
    print(prob)
    