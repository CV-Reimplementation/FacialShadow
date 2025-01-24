from data import get_validation_data
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

    os.makedirs(os.path.join(opt.MODEL.SESSION, 'UCB'), exist_ok=True)
    os.makedirs(os.path.join(opt.MODEL.SESSION, 'ASFW'), exist_ok=True)

    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    # Data Loader
    val_file1 = opt.TRAINING.VAL_FILE1
    val_file2 = opt.TRAINING.VAL_FILE2

    val_dataset1 = get_validation_data(val_file1, None)
    val_dataset2 = get_validation_data(val_file2, None)

    testloader1 = DataLoader(dataset=val_dataset1, batch_size=1, shuffle=False, num_workers=16, drop_last=False, pin_memory=True)
    testloader2 = DataLoader(dataset=val_dataset2, batch_size=1, shuffle=False, num_workers=16, drop_last=False, pin_memory=True)

    # Model & Metrics
    model = DeShadowNet()

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader1, testloader2 = accelerator.prepare(model, testloader1, testloader2)

    model.eval()

    size1 = len(testloader1)
    size2 = len(testloader2)

    stat_psnr = 0
    stat_ssim = 0
    stat_lpips = 0
    stat_rmse = 0

    for _, test_data in enumerate(tqdm(testloader1)):
        # get the inputs; data is a list of [targets, inputs, filename]
        inp = test_data[0].contiguous()
        tar = test_data[1]
        mas = mask_generator(inp, tar)

        with torch.no_grad():
            res = model(inp, mas).clamp(0, 1)

        save_image(res, os.path.join(opt.MODEL.SESSION, 'UCB', test_data[2][0]))

        stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
        stat_ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
        stat_lpips += criterion_lpips(res, tar).item()
        stat_rmse += mean_squared_error(torch.mul(res, 255).flatten(), torch.mul(tar, 255).flatten(), squared=False).item()

    stat_psnr /= size1
    stat_ssim /= size1
    stat_lpips /= size1
    stat_rmse /= size1

    print("RMSE: {}, PSNR: {}, SSIM: {}, LPIPS: {}".format(stat_rmse, stat_psnr, stat_ssim, stat_lpips))

    stat_psnr = 0
    stat_ssim = 0
    stat_lpips = 0
    stat_rmse = 0

    for _, test_data in enumerate(tqdm(testloader2)):
        # get the inputs; data is a list of [targets, inputs, filename]
        inp = test_data[0].contiguous()
        tar = test_data[1]
        mas = mask_generator(inp, tar)

        with torch.no_grad():
            res = model(inp, mas).clamp(0, 1)

        save_image(res, os.path.join(opt.MODEL.SESSION, 'ASFW', test_data[2][0]))

        stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
        stat_ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
        stat_lpips += criterion_lpips(res, tar).item()
        stat_rmse += mean_squared_error(torch.mul(res, 255).flatten(), torch.mul(tar, 255).flatten(), squared=False).item()

    stat_psnr /= size2
    stat_ssim /= size2
    stat_lpips /= size2
    stat_rmse /= size2

    print("RMSE: {}, PSNR: {}, SSIM: {}, LPIPS: {}".format(stat_rmse, stat_psnr, stat_ssim, stat_lpips))


if __name__ == '__main__':
    test()
