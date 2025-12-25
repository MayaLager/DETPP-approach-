class GRU4Rec(nn.Module):
    def __init__(self, n_items, embed_dim, hidden_size, dropout):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_items + 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.item_emb(x))
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class SASRec(nn.Module):
    def __init__(self, n_items, embed_dim, n_heads, n_layers, dropout, max_len=SEQ_LEN):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=embed_dim * 2,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.out = nn.Linear(embed_dim, n_items + 1)

    def forward(self, x):
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.dropout(self.item_emb(x) + self.pos_emb(pos))
        h = self.transformer(h)
        return self.out(h[:, -1, :])

class T2ARec(SASRec):
    def __init__(self, n_items, embed_dim, n_heads, n_layers, dropout):
        super().__init__(n_items, embed_dim, n_heads, n_layers, dropout)

    def time_interval_loss(self, seq):
        return torch.tensor(0.0, device=seq.device)

    def state_alignment_loss(self, rep):
        return torch.tensor(0.0, device=rep.device)

    def forward_with_ssl(self, x):
        logits = super().forward(x)
        L_time = self.time_interval_loss(x.float())
        L_state = self.state_alignment_loss(x)
        return logits, L_time, L_state