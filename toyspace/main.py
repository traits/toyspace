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

# TODO https://discuss.pytorch.org/t/dataloader-returns-cpu-tensors/17933/2
device = torch.device(
    f"cuda:{cuda.Cuda().device_count()-1}" if torch.cuda.is_available() else "cpu"
)


if __name__ == "__main__":
    decision_regions_img = loadToyImage(data_dir / "randomborder.png")
    decision_regions_img = cv2.normalize(
        decision_regions_img, decision_regions_img, 1, 255, cv2.NORM_MINMAX
    )
    # samples = selectSample(decision_regions_img, ROI_sampler, [25, 32, 160, 177])

    samples = selectSample(decision_regions_img, partition_sampler)

    # random 5% state space coverage
    samples = selectSample(
        decision_regions_img, random_sampler, image_area(decision_regions_img) // 10
    )
    samples_img = np.zeros(decision_regions_img.shape, dtype=np.uint8)
    for p in samples:
        samples_img[p[1], p[2]] = p[0]

    # https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    mplot_map = "PuBuGn"
    write_image(
        out_dir / "randomborder_samples.png", colormapped_image(samples_img, mplot_map)
    )
    write_image(
        out_dir / "randomborder_colored.png",
        colormapped_image(decision_regions_img, mplot_map),
    )
    labels = samples[:, 0]
    coords = samples[:, 1:]

    train_data = convert2Dataset(coords, labels, device)
    #     # save_as_png(images, out_dir, correct_labels=labels, pred_labels=None)
    #     print("Trainings data extracted")
    #     test_images = loadDataAsDataset(data_dir / "test.csv", device)
    #     print("Test data extracted")
    #
    model = SimpleNet()
    train(model, train_data, epochs=300, bs=20, lr=0.008, device=device)
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
