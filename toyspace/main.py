from pytraits.image.base import *
from pytraits.image.io import *
from pytraits.pytorch import cuda


from data import *
from models import *
from process import *

from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
out_dir = project_dir / "_output"

# TODO https://discuss.pytorch.org/t/dataloader-returns-cpu-tensors/17933/2
device = torch.device(
    f"cuda:{cuda.Cuda().device_count()-1}" if torch.cuda.is_available() else "cpu"
)


if __name__ == "__main__":

    x0 = 25
    x1 = 160
    y0 = 32
    y1 = 177

    def sampler(img):
        roi = img[y0:y1, x0:x1]
        samples = roi.copy()
        y, x = np.indices(roi.shape)
        y += y0
        x += x0
        samples = np.dstack((samples, y, x))
        s = samples.shape
        return samples.reshape(s[0] * s[1], s[2])

    img = loadToyImage(data_dir / "randomborder.png")
    samples = selectSample(img, sampler)
    img2 = np.zeros(img.shape)
    for p in samples:
        img2[p[1], p[2]] = p[0]
    write_image(out_dir / "randomborder.png", img2)

    # for p in samples

    #     train_data = convert2Dataset(images, device)
    #     # save_as_png(images, out_dir, correct_labels=labels, pred_labels=None)
    #     print("Trainings data extracted")
    #     test_images = loadDataAsDataset(data_dir / "test.csv", device)
    #     print("Test data extracted")
    #
    #     # model = ConvNet()
    #     # pred_labels = run(model, train_data, epochs=1, bs=500, lr=0.008, test_data=test_images, device=device)
    #
    #     nets = createBinaryEnsemble()
    #     size = len(nets)
    #     for i in range(size):
    #         binlabels = BinaryClassifier.ajustLabels(labels, i)
    #         print(f"starting {i}-classifier [{i}/{size-1}]")
    #         train(
    #             nets[i],
    #             convert2Dataset(images, binlabels, device),
    #             epochs=1,
    #             bs=1000,
    #             lr=0.0085,
    #             device=device,
    #         )
    #     print("finished training")
    #     pred_labels = testEnsemble(nets, test_images, device=device)
    #     print("finished test")

    print("script completed")
