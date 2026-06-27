import matplotlib.pyplot as plt
import kornia as K
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches


def visualize_matches(
    img1,
    img2,
    kps1,
    kps2,
    idxs,
    inliers=None,
    output_path="/results",
):
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(kps1[None].cpu()),
        KF.laf_from_center_scale_ori(kps2[None].cpu()),
        idxs.cpu(),
        K.tensor_to_image(img1.cpu()),
        K.tensor_to_image(img2.cpu()),
        inliers,
        draw_dict={
            "inlier_color": (0.2, 1, 0.2),
            "tentative_color": (1, 1, 0.2, 0.3),
            "feature_color": None,
            "vertical": False,
        },
    )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")

    plt.show()

def visualize_keypoints(image, keypoints, output_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(K.tensor_to_image(image.cpu()))

    ax.scatter(
        keypoints[:, 0].cpu(),
        keypoints[:, 1].cpu(),
        s=2
    )

    ax.set_axis_off()

    if output_path:
        plt.savefig(output_path, dpi=300)

    plt.show()