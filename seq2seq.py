import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers, dropout_ratio):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout_ratio)

        self.dropout = nn.Dropout(dropout_ratio)

          
    def forward(self, src):
        # src: [len_word, batch_size]: 각 단어 index
        embedded = self.dropout(self.embedding(src))
        # embedded: [len_word, batch_size, embedding_dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [단어 개수, batch_size, hidden_dim]: 현재 단어의 출력 정보
        # hidden: [num_layers, batch_size, hidden_dim]: 현재까지 모든 단어 정보
        # cell: [num_layers], batch_size, hidden_dim]: 현재까지 모든 단어 정보

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, dropout_ratio):
        super().__init__()

        self.embedding = nn.Embedding(output_dim, embed_dim)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout_ratio)

        self.output_dim = output_dim
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout_ratio)
   
    def forward(self, input, hidden, cell):
        # input: [배치 크기]: 단어의 개수는 항상 1개이도록 구현
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell = context: [num_layers, batch_size, hidden_dim]
        input = input.unsqueeze(0)
        # input: [단어 개수 = 1, batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded: [단어 개수, batch_size, embedding_dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [단어 개수 = 1, batch_size, hidden_dim]: 현재 단어의 출력 정보
        # hidden: [num_layers, batch_size, hidden_dim]: 현재까지 모든 단어 정보
        # cell: [num_layers, batch_size, hidden_dim]: 현재까지 모든 단어 정보

        # 단어 개수는 1개이므로 차원 제거
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        
        # (현재 출력 단어, 현재까지 모든 단어 정보, 현재까지 모든 단어 정보)
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [len_word, batch_size]
        # trg: [len_word, batch_size]
        hidden, cell = self.encoder(src)

        # decoder 최종 결과를 담을 텐서 객체 만들기
        # len_word
        trg_len = trg.shape[0] 

        batch_size = trg.shape[1] 
        # output_dim
        trg_vocab_size = self.decoder.output_dim 
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 첫 번째 입력 sos 토큰
        input = trg[0, :]

        # target 단어 개수 반복
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)

            # 현재의 출력 단어 정보
            outputs[t] = output 
            # 가장 확률이 높은 단어 인덱스 
            top1 = output.argmax(1) 

            # teacher_forcing_ratio: 학습 시 ground-truth 사용 비율
            teacher_force = random.random() < teacher_forcing_ratio
            # 현재 output을 다음 입력에 넣음
            input = trg[t] if teacher_force else top1 
        
        return outputs