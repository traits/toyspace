from pytraits.image.region import *
from pytraits.pytorch import cuda


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

if __name__ == "__main__":
    regions, categories = load_toy_image(toy_image)
    write_color_image(maximize_image(regions), out_dir / "randomborder_colored.png")

    samples = random_sampler(regions, image_area(regions) // 20)
    write_sample_image(
        regions.shape, categories, samples, out_dir / "random_sampler_all.png"
    )

    labels = samples[:, 0]
    coords = np.multiply(samples[:, 1:], 1.0 / 255.0)  # normalize in [0,1]

    train_data = convert_to_dataset(coords, labels, device)
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
