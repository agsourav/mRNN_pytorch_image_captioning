import torch
import torch.nn as nn

class mRNN(nn.Module):
    def __init__(self, inp_dim, hidden_dim):
        super(mRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(inp_dim+hidden_dim, hidden_dim, num_layers = 1, batch_first = False, 
                            nonlinearity = 'relu')
        self.Ur = nn.Linear(hidden_dim, hidden_dim)
        self.fr = nn.ReLU()

    def forward(self, wt, rt_1):
        wt = torch.cat([wt, rt_1], dim = 2)        # current word input
        wt1, rt = self.rnn(wt, rt_1)
        rt = self.Ur(rt.squeeze(0))
        rt = self.fr(rt.unsqueeze(0) + wt1)    #[batch_size, hidden_dim]

        return rt  # present rnn state

class AlexNet(nn.Module):
    def __init__(self, feature_vector_dim = 128):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64, kernel_size = 11, stride = 4, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(64, 192, kernel_size = 5, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.Linear(1024, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, feature_vector_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ImageFeatures(object):
    def __init__(self):
        super(ImageFeatures, self).__init__()
        self.alexnet = AlexNet()

    def load_weights(self, weight_path = 'data/weight/alexnet_custom.pth'):
        try:
            state_dict = torch.load(weight_path)
        except:
            print('Could not load file')
            exit()
        self.alexnet.load_state_dict(state_dict)


    def feature_vector(self, x):
        self.load_weights()
        feature_v = self.alexnet(x)
        return feature_v
        
class Model(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim, feature_dim):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed2 = nn.Linear(input_dim, embed_dim)
        self.rnn = mRNN(embed_dim, hidden_dim)
        self.imagefeatures = ImageFeatures()
        self.Vr = nn.Linear(hidden_dim, hidden_dim)
        self.Vw = nn.Linear(embed_dim, hidden_dim)
        self.Vi = nn.Linear(feature_dim, hidden_dim)
        self.G = nn.Tanh()

    def forward(self, x, img, max_caption_length):   #embedded representation of word sequence [seq_len, batch, embedding_dim]
        batch_size = x.shape[1]
        w0 = self.embed2(x.squeeze(0))  #[batch, embed_dim]
        h0 = torch.zeros((1, batch_size, self.hidden_dim))
        # w0: [1, batch_size, embed_dim]    h0: [1, batch_size, hidden_dim]
        wt = w0
        caption = wt.unsqueeze(0)
        rt_1 = h0
        for i in range(max_caption_length):
            wt = wt.unsqueeze(0)
            rt = self.rnn(wt, rt_1)  
        # wt: [1, batch_size, embed_dim]    rt: [1, batch_size, hidden_dim]
            Vwt = self.Vw(wt.squeeze(0))
            Vrt = self.Vr(rt.squeeze(0))

            feature_vector_img = self.imagefeatures.feature_vector(img)
            Vit = self.Vi(feature_vector_img)
            
            mt = Vwt + Vrt + Vit    # [batch_size, feature_dim]
            mt = self.G(mt)         # [batch_size, feature_dim]
            rt_1 = rt
            wt = mt
            caption = torch.cat([caption, wt.unsqueeze(0)], dim = 0)
            
        return caption

