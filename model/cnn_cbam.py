import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# def convmixer_layer(dim, depth, inside_dim, kernel_size = 9):
#     return nn.Sequential(
#         *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
#                     nn.GELU(),
#                     nn.BatchNorm2d(dim)
#                 )),
#                 nn.Conv2d(dim, dim, kernel_size=1),
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim)
#         ) for i in range(depth)],
#     )

def convmixer_layer(dim, depth, inside_dim, kernel_size):
    return nn.Sequential(
        *[nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(in_channels = dim, out_channels = inside_dim, kernel_size = 1, stride = 1, padding = 0),
                        nn.GELU(),
                        # nn.LeakyReLU(),

                        nn.Conv2d(in_channels = inside_dim, out_channels = inside_dim, kernel_size = kernel_size, stride = 1, padding = 1),
                        nn.GELU(),
                        # nn.LeakyReLU(),
                        
                        nn.Conv2d(in_channels = inside_dim, out_channels = dim, kernel_size = 1, stride = 1, padding = 0),
                        nn.GELU(),
                        # nn.LeakyReLU(),
                        )
                    ),
                # nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                # nn.LeakyReLU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
    )

class CnnCbam(nn.Module):
    def __init__(self, dim = 768, depth = 24, inside_dim = 64, kernel_size = 3, patch_size = 16, n_classes = 2):
        super(CnnCbam, self).__init__()
        self.dim = dim
        self.depth = depth
        self.n_class = n_classes
        self.inside_dim = inside_dim
        self.kernel_size = kernel_size

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size = patch_size, stride = patch_size),
            nn.GELU(),
            # nn.LeakyReLU(),
            nn.BatchNorm2d(dim)
        )

        self.cm_layer = convmixer_layer(self.dim, self.depth, self.inside_dim, self.kernel_size)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(dim, n_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cm_layer(x)
        x = self.gap(x)
        x = self.flat(x)
        output = self.fc(x)
        return output

if __name__ == '__main__':
    import torch
    import netron
    import torch.optim as optim
    from cnn_cbam import CnnCbam


    # GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)

    model = CnnCbam()
    print(model)

    _input = torch.randn(1, 3, 640, 640)
    # _input = _input.unsqueeze(0)
    # print(_input.size())
    onxx_path = r'G:\我的雲端硬碟\Lab\Project\胸大肌\乳腺\Breast_PL\model.onnx'
    torch.onnx.export(model, _input, onxx_path)
    netron.start(onxx_path)