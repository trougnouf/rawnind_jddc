import torch
import sys

sys.path.append("..")
from rawnind.libs import raw

assert len(sys.argv) == 3
imgt1, _ = raw.raw_fpath_to_rggb_img_and_metadata(sys.argv[1])
imgt2, _ = raw.raw_fpath_to_rggb_img_and_metadata(sys.argv[2])
print(torch.nn.MSELoss()(torch.tensor(imgt1), torch.tensor(imgt2)).item())
