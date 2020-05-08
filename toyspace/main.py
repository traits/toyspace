from pytraits.image.region import *
from pytraits.image.base import *
from pytraits.image.io import *
from pytraits.pytorch import cuda
import cv2


from data import *
from models import *
from process import *

from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
out_dir = project_dir / "_output"

toy_image = data_dir / "randomborder.png"


# TODO https://discuss.pytorch.org/t/dataloader-returns-cpu-tensors/17933/2
device = torch.device(
    f"cuda:{cuda.Cuda().device_count()-1}" if torch.cuda.is_available() else "cpu"
)


def maximizedImage(img):
    result = img.copy()
    # preserve zero for background
    return cv2.normalize(img, result, 1, 255, cv2.NORM_MINMAX)


def writeColorImage(img, fname):
    # https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    mplot_map = "PuBuGn"
    write_image(
        out_dir / fname, colormapped_image(img, mplot_map),
    )


def writeSampleImage(shape, max_value, samples, fname):
    samples_img = np.zeros(shape, dtype=np.uint8)
    for p in samples:
        samples_img[p[1], p[2]] = np.uint8(255.0 * p[0] / max_value)

    writeColorImage(samples_img, fname)


if __name__ == "__main__":
    regions, categories = loadToyImage(toy_image)
    writeColorImage(maximizedImage(regions), "randomborder_colored.png")

    # samples = selectSample(regions, ROI_sampler, [25, 32, 160, 177])
    # samples = selectSample(regions, partition_sampler)
    # writeSampleImage(regions.shape, categories, samples[3], "randomborder_samples.png")

    # samples = selectSample(regions, grid_sampler, [20, 10], [80, 120, 150, 190])
    # random 5% state space coverage
    # samples = selectSample(regions, random_sampler, image_area(regions) // 20)

    regions_g = ((255 / categories) * regions).astype(np.uint8)
    contours, _ = find_contours(regions_g, 100, complexity=cv2.RETR_EXTERNAL)
    cv2.drawContours(regions, contours, -1, 255, 1)
    cv2.imwrite(str(out_dir / "test.png"), regions)

    samples = selectSample(regions, random_sampler2, contours[2], 100)
    writeSampleImage(regions.shape, categories, samples, "randomborder_samples.png")

    labels = samples[:, 0]
    coords = np.multiply(samples[:, 1:], 1.0 / 255.0)  # normalize in [0,1]

    train_data = convert2Dataset(coords, labels, device)
    #     # save_as_png(images, out_dir, correct_labels=labels, pred_labels=None)
    #     print("Trainings data extracted")
    #     test_images = loadDataAsDataset(data_dir / "test.csv", device)
    #     print("Test data extracted")
    #
    model = SimpleNet()
    train(model, train_data, epochs=300, bs=17, lr=0.008, device=device)
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
