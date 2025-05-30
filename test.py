from data import get_data
import warnings

warnings.filterwarnings('ignore')

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.regression import mean_squared_error
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from models import *
from utils import *


def test():
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator()
    device = accelerator.device

    os.makedirs(opt.MODEL.SESSION, exist_ok=True)

    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    # Data Loader
    val_file = opt.TRAINING.VAL_FILE

    val_dataset = get_data(val_file, opt.MODEL.INPUT, opt.MODEL.TARGET, 'test', opt.TRAINING.ORI,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})

    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=16, drop_last=False, pin_memory=True)

    # Model & Metrics
    model = model_registry.get(opt.MODEL.SESSION)()
    
    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader = accelerator.prepare(model, testloader)

    model.eval()

    size = len(testloader)

    stat_psnr = 0
    stat_ssim = 0
    stat_lpips = 0
    stat_rmse = 0

    for _, test_data in enumerate(tqdm(testloader)):
        # get the inputs; data is a list of [targets, inputs, filename]
        inp = test_data[0].contiguous()
        tar = test_data[1]
        mas = test_data[3]

        with torch.no_grad():
            res = model(inp, mas).clamp(0, 1)

        save_image(res, os.path.join(opt.MODEL.SESSION, test_data[2][0]))

        stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
        stat_ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
        stat_lpips += criterion_lpips(res, tar).item()
        stat_rmse += mean_squared_error(torch.mul(res, 255).flatten(), torch.mul(tar, 255).flatten(), squared=False).item()

    stat_psnr /= size
    stat_ssim /= size
    stat_lpips /= size
    stat_rmse /= size

    print("RMSE: {}, PSNR: {}, SSIM: {}, LPIPS: {}".format(stat_rmse, stat_psnr, stat_ssim, stat_lpips))


if __name__ == '__main__':
    test()
