import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torchvision.utils import save_image

from dataset import build_dataset_from_args, create_dataloader
from model import UNET_MODEL


def parse_args():
    parser = argparse.ArgumentParser(description="UNet segmentation evaluation")

    parser.add_argument("--test-image-dir", type=str, required=True)
    parser.add_argument("--test-mask-dir", type=str, required=True)

    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="./test_results")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--image-height", type=int, default=448)
    parser.add_argument("--image-width", type=int, default=448)

    parser.add_argument("--channel-reduction", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def compute_batch_iou(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    iou = intersection / np.maximum(union.astype(np.float32), 1e-8)
    return float(np.mean(iou))


def compute_batch_f1(y_true, y_pred, eps=1e-8):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    tp = np.sum(y_true * y_pred)
    fp = np.sum((1.0 - y_true) * y_pred)
    fn = np.sum(y_true * (1.0 - y_pred))

    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2.0 * precision * recall / max(precision + recall, eps)

    return float(f1)


def load_model(args, device):
    model = UNET_MODEL(
        args=None,
        channel_reduction=args.channel_reduction,
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def save_predictions(predictions, results_dir, start_index):
    results_dir.mkdir(parents=True, exist_ok=True)

    image_index = start_index
    for i in range(predictions.shape[0]):
        output_path = results_dir / f"result_{image_index:04d}.png"
        save_image(predictions[i], str(output_path))
        image_index += 1

    return image_index


@torch.no_grad()
def evaluate_model(model, dataloader, device, threshold, results_dir):
    total_f1 = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_iou = 0.0
    total_batches = 0
    image_counter = 0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()

        image_counter = save_predictions(predictions, results_dir, image_counter)

        y_true = masks.cpu().numpy().astype(np.uint8).flatten()
        y_pred = predictions.cpu().numpy().astype(np.uint8).flatten()

        batch_f1 = compute_batch_f1(y_true, y_pred)
        batch_recall = recall_score(y_true, y_pred, zero_division=0)
        batch_precision = precision_score(y_true, y_pred, zero_division=0)
        batch_iou = compute_batch_iou(y_true, y_pred)

        total_f1 += batch_f1
        total_recall += batch_recall
        total_precision += batch_precision
        total_iou += batch_iou
        total_batches += 1

    return {
        "f1": total_f1 / total_batches,
        "recall": total_recall / total_batches,
        "precision": total_precision / total_batches,
        "miou": total_iou / total_batches,
        "saved_dir": str(results_dir),
    }


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    args.test_image_dir = args.test_image_dir
    args.test_mask_dir = args.test_mask_dir

    test_dataset = build_dataset_from_args(args, "test")
    test_dataloader = create_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = load_model(args, device)
    results_dir = Path(args.results_dir)

    metrics = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=device,
        threshold=args.threshold,
        results_dir=results_dir,
    )

    print(f"Test F1 Score: {metrics['f1']:.4f}")
    print(f"Test Recall Score: {metrics['recall']:.4f}")
    print(f"Test Precision Score: {metrics['precision']:.4f}")
    print(f"Test mIoU Score: {metrics['miou']:.4f}")
    print(f"Saved binary results to: {metrics['saved_dir']}")


if __name__ == "__main__":
    main()