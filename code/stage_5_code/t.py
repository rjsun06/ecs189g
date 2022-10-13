import torch.utils.data
from torch import optim, nn
import pandas as pd
import numpy as np
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# ---- Multi-Layer Perceptron script ----

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device='cpu'
num_time_steps = 34
input_size = 64
hidden_size = 16
output_size = 1
vocab_size=0
embedding_dim=64

voc=dict.fromkeys([])
voci=dict.fromkeys([])

def MyRead(path):
    count = 1
    set=[]
    with open(path,'rb') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line=str(line)
            #print(line)
            i=1
            while(line[i]!=','):
                #print(line[i])
                i=i+1
            newline=[]
            #print(line[i+1:-4].replace('"','').replace('\'','').replace('?','').replace('!','').replace(',','').replace('.','').replace('[','').replace(']','').replace('\\',''))
            for word in line[i+1:-4].lower().replace('"','').replace('\'','').replace('?','').replace('!','').replace(',','').replace('.','').replace('[','').replace(']','').replace('\\','').split(' '):
                voc.setdefault(word,count)
                if voc[word]==count:
                    voci[count]=word
                    count+=1
                newline.append(voc[word])

            set.append(newline)

    return set


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.embed=nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.liner = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        print("in",x)
        emb=self.embed(x)
        print("emb",emb)
        #print(emb)
        out, hidden_prev = self.rnn(emb, hidden_prev)
        print("out",out)
        out = out.view(-1, hidden_size)
        out = self.liner(out)
        out = out.unsqueeze(dim=0)
        print("fin",out)
        return out, hidden_prev



train = MyRead('C:/cygwin64/home/17931/workspace/ecs189G/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_generation/data')
vocab_size=len(voc.keys())

test = [[voc['what'], voc['did'], voc['the']],[voc['what'], voc['kind'], voc['of']]]

model = Net().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), 1e-2)

hidden_prev = torch.zeros( 1,1, hidden_size).to(device)








ltest = np.array(test).tolist()
ltrain = np.array(train).tolist()
#print(ltrain)
train=[]
for i in ltrain:
    for j in range(0, len(i) - 4):
        train.append(i[j:j+4])
x_train = torch.tensor([i[0:3] for i in train]).to(device)
y_train = torch.tensor([i[1:4] for i in train]).float().to(device)
print(x_train)
print(y_train)
x_test = torch.tensor(test).to(device)

print(x_train)



for iter in range(10):
  for i in range(10):

        output, hidden_prev = model(x_train[i:i+1], hidden_prev)
        hidden_prev = hidden_prev.detach()

        loss = criterion(output, y_train[i])
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10000 == 0:
          print("Iter: {} loss: {} ".format(iter, loss))
  torch.save(model, 'model'+str(iter)+'.pth')




print("pred Start")

predict = []

for i in range(len(x_test)):
      raw = x_test[i:i+1]
      input = raw

      for _ in range(34):
          print(input)
          (pred, hidden_prev) = model(input.long(), hidden_prev)
          print(pred)
          print(input[0][1:], pred[0][2], "jj")
          input[0] = input[0][1:] + pred[0][2]

          predict.append(pred)
print(predict)
for i in predict:
      for j in i:
          print(voci[int(j)],end=' ')