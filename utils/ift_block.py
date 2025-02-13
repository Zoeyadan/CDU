from einops import rearrange
import torch.nn as nn

class IFT_Module(nn.Module):
    """ IFT """

    def __init__(self, tea_dim=768
                ):
        super().__init__()

        self.softmax = nn.Softmax(-1)
        input_dim = tea_dim
        pre_dim1 = input_dim // 8
        pre_dim2 = input_dim // 8

        self.scale = 0.1

        self.pre_project = nn.Sequential(  # 3 layers
            nn.Linear(input_dim, pre_dim1),
            nn.BatchNorm1d(pre_dim1),
            nn.ReLU(inplace=True),

            nn.Linear(pre_dim1, pre_dim2),
            nn.BatchNorm1d(pre_dim2),
            nn.ReLU(inplace=True),

            nn.Linear(pre_dim2, input_dim * 3)
        ).half()

        self.post_project = nn.Sequential(  # only one layer
            nn.Linear(input_dim, input_dim)
        ).half()


    def forward(self, Fv, Fvs_bank, Fvt_bank):
        '''
        Fvs with shape (batch, C): source visual output w/o attnpool
        Fvt with shape (N, C): classes of target visual output w/o attnpool
        '''
        out_fv = self.pre_project(Fv)  # (batch, 3 * C)
        out_fvs = self.pre_project(Fvs_bank)  # (N, 3 * C)
        out_fvt = self.pre_project(Fvt_bank)  # (N, 3 * C)

        q_fv, k_fv, v_fv = tuple(rearrange(out_fv, 'b (d k) -> k b d ', k=3))
        q_fvs, k_fvs, v_fvs = tuple(rearrange(out_fvs, 'b (d k) -> k b d ', k=3))
        q_fvt, k_fvt, v_fvt = tuple(rearrange(out_fvt, 'b (d k) -> k b d ', k=3))

        As = self.softmax(self.scale * q_fv @ k_fvs.permute(1, 0))  # (batch, N)
        At = self.softmax(self.scale * q_fv @ k_fvt.permute(1, 0))  # (batch, N)

        Fsa = Fv + self.post_project(As @ v_fvs)  # (batch, C)
        Fta = Fv + self.post_project(At @ v_fvt)  # (batch, C)

        Fsa = Fsa / Fsa.norm(dim=-1, keepdim=True)
        Fta = Fta / Fta.norm(dim=-1, keepdim=True)


        return Fsa, Fta