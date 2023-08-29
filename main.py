from pathlib import Path

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from Abel import abel_inversion


def main():
    img_path = Path(__file__).parent / "Img"
    axis = 199
    img_file = img_path / "Sample_1.tif"

    img = np.array(Image.open(img_file), np.int16)

    axis_symmetrized_img = (img[:,axis:] + img[:,:axis + 1][:,::-1]) / 2

    abel_img = abel_inversion(axis_symmetrized_img)

    plt.imshow(np.maximum(abel_img, 0), cmap="jet")
    plt.show()


if __name__ == "__main__":
    main()