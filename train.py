from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    dropout: float = 0.1
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.c_attn = nn.Linear(self.n_embd, 3*self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B, T, C = x.size() # [B, T, n_embd]
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # [B, T, n_embd] each
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, n_embd//n_head]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, n_embd//n_head]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, n_embd//n_head]
        
        attn = (q @ k.transpose(-2, -1)) * 1.0 / (k.size(-1)**0.5) # [B, n_head, T, T]
        attn = F.softmax(attn, dim=-1)  # [B, n_head, T, T]
        attn = self.attn_dropout(attn)   # [B, n_head, T, T]
        
        y = attn @ v    # [B, n_head, T, n_embd//n_head]
        
        y = y.transpose(1,2).contiguous().view(B, T, C) # [B, T, n_embd]
        y = self.c_proj(y)  # [B, T, n_embd]
        y = self.res_dropout(y) # [B, T, n_embd]
        
        return y        


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x) # [B, T, 4*n_embd]
        x = F.gelu(x)   # [B, T, 4*n_embd]
        x = self.c_proj(x)  # [B, T, n_embd]
        x = self.drop(x)    # [B, T, n_embd]
        return x
    
    
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx):
        B, T = idx.size()
        
        pos = torch.arange(T, dtype=torch.long, device=idx.device).unsqueeze(0) # [1, T]
        
        wte = self.wte(idx) # [B, T, n_embd]
        wpe = self.wpe(pos) # [1, T, n_embd]
        
        x = self.dropout(wte+wpe) # [B, T, n_embd]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x) # [B, T, n_embd]
        
        x = self.ln_f(x) # [B, T, n_embd]
        logits = self.lm_head(x) # [B, T, vocab_size]
        
        return logits


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0
        self.current_epoch = 0  # Track the current epoch
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
            self.current_epoch += 1  # Increment the epoch count
            
        return x, y


def generate_sequences(model, enc, num_return_sequences=5, max_length=30, device='cpu'):
    # x = torch.zeros((num_return_sequences, 1), dtype=torch.long, device=device)  # Initialize with a start token or zeros
    prompt = "I tell "
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits = model(x)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


if __name__ == "__main__":
    # SEED
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")

    config = GPTConfig()
    model = GPT(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    train_loader = DataLoaderLite(B = 4, T = 128)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)

    num_epochs = 140  # Set the number of epochs
    for epoch in range(num_epochs):  # Loop over the specified number of epochs
        stop = False
        num_batches = len(train_loader.tokens) // (train_loader.B * train_loader.T)  # Total number of batches

        # Initialize tqdm instance
        progress_bar = tqdm(range(num_batches), desc=f'Epoch {epoch + 1}, Loss: 0.0000')  # Initial description

        for _ in progress_bar:  # Use the tqdm instance
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)  # Forward pass
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            # Update tqdm description with the current loss
            progress_bar.set_description(f'Epoch {epoch + 1}, loss/seq: {loss.item():.4f}')  # Update the tqdm description
            
            if loss.item() < 0.099999: # coz i raised the seq_len in dataloader by 4x
                stop = True
                break
            
            torch.cuda.empty_cache()
            del x, y, logits, loss
            
        if stop:
            print('Stopping training...')
            torch.save(model.state_dict(), f'model.pt')
            print("Checkpoint saved...")
            break
        
        # Generate sequences, save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Generating sequences after epoch {epoch + 1}:")
            generate_sequences(model, tiktoken.get_encoding('gpt2'), device=device)
            torch.save(model.state_dict(), f'model.pt')
            print("Checkpoint saved...")
            print("#"*30)

    print("Training completed...")