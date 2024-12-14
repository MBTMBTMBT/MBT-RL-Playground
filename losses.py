import torch
import torch.nn as nn


# Enhanced loss with separate handling for L1 and L2 (MSE) losses with optional thresholding
class FlexibleThresholdedLoss(nn.Module):
    def __init__(self, use_mse_threshold=False, use_mae_threshold=True, mse_threshold=None, mae_threshold=None,
                 reduction='mean', l1_weight=1.0, l2_weight=1.0, threshold_weight=1.0, non_threshold_weight=1.0,
                 mse_clip_ratio=None, mae_clip_ratio=1e1):
        """
        use_mse_threshold: Whether to apply a threshold to L2 (MSE)-based loss.
        use_mae_threshold: Whether to apply a threshold to L1-based loss.
        mse_threshold: Static L2 (MSE)-based threshold if provided; otherwise, will use adaptive MSE threshold.
        mae_threshold: Static L1-based threshold if provided; otherwise, will use adaptive MAE threshold.
        reduction: Specifies the reduction to apply to the output. ('mean' or 'sum').
        l1_weight: The weight for L1 loss in both thresholded and non-thresholded parts.
        l2_weight: The weight for L2 (MSE) loss in both thresholded and non-thresholded parts.
        threshold_weight: Weight for the thresholded loss part.
        non_threshold_weight: Weight for the non-thresholded loss part.
        mse_clip_ratio: Ratio to apply the upper limit clipping for MSE thresholded loss.
        mae_clip_ratio: Ratio to apply the upper limit clipping for MAE thresholded loss.
        """
        super(FlexibleThresholdedLoss, self).__init__()
        self.use_mse_threshold = use_mse_threshold
        self.use_mae_threshold = use_mae_threshold
        self.mse_threshold = mse_threshold
        self.mae_threshold = mae_threshold
        self.reduction = reduction
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.threshold_weight = threshold_weight
        self.non_threshold_weight = non_threshold_weight
        self.mse_clip_ratio = mse_clip_ratio
        self.mae_clip_ratio = mae_clip_ratio

    def forward(self, input_img, target_img):
        # Calculate pixel-wise absolute difference (for L1) and squared difference (for L2/MSE)
        pixel_diff = torch.abs(input_img - target_img)  # For L1
        pixel_diff_squared = (input_img - target_img) ** 2  # For L2 (MSE)

        # General mean for normalization
        general_mean = pixel_diff.mean()

        # Part 1: L2 (MSE)-based threshold handling with optional normalization
        mse_loss = pixel_diff_squared.mean()  # MSE (L2) loss
        if self.use_mse_threshold:
            if self.mse_threshold is None:
                self.mse_threshold = mse_loss  # Set adaptive threshold based on MSE

            # Filter values based on threshold
            mse_thresholded_diff = pixel_diff_squared[pixel_diff_squared >= self.mse_threshold]

            # Calculate the thresholded loss and normalize if necessary
            if mse_thresholded_diff.numel() > 0:
                mse_thresholded_loss = mse_thresholded_diff.mean()
                if self.mse_clip_ratio is not None:
                    clip_value = self.mse_clip_ratio * general_mean
                    # Normalize the loss if it exceeds the clip_value
                    if mse_thresholded_loss > clip_value:
                        mse_thresholded_loss = clip_value * mse_thresholded_loss / mse_thresholded_loss.detach()
            else:
                mse_thresholded_loss = torch.tensor(0.0, device=pixel_diff.device)
        else:
            # No thresholding, use all squared differences for L2 (MSE)
            mse_thresholded_loss = pixel_diff_squared.mean()

        # Part 2: L1-based threshold handling with optional normalization
        mae_loss = pixel_diff.mean()  # L1 (absolute difference) loss
        if self.use_mae_threshold:
            if self.mae_threshold is None:
                self.mae_threshold = mae_loss  # Set adaptive threshold based on MAE

            # Filter values based on threshold
            mae_thresholded_diff = pixel_diff[pixel_diff >= self.mae_threshold]

            # Calculate the thresholded loss and normalize if necessary
            if mae_thresholded_diff.numel() > 0:
                mae_thresholded_loss = mae_thresholded_diff.mean()
                if self.mae_clip_ratio is not None:
                    clip_value = self.mae_clip_ratio * general_mean
                    # Normalize the loss if it exceeds the clip_value
                    if mae_thresholded_loss > clip_value:
                        mae_thresholded_loss = clip_value * mae_thresholded_loss / mae_thresholded_loss.detach()
            else:
                mae_thresholded_loss = torch.tensor(0.0, device=pixel_diff.device)
        else:
            # No thresholding, use all absolute differences for L1
            mae_thresholded_loss = pixel_diff.mean()

        # Part 3: Non-thresholded loss (all differences are considered for both L1 and L2)
        non_thresholded_l1_loss = pixel_diff.mean()  # L1 part without threshold
        non_thresholded_l2_loss = pixel_diff_squared.mean()  # L2 (MSE) part without threshold

        # Combine thresholded L1 and L2 losses
        combined_thresholded_loss = self.l1_weight * mae_thresholded_loss + self.l2_weight * mse_thresholded_loss

        # Combine non-thresholded L1 and L2 losses
        combined_non_thresholded_loss = self.l1_weight * non_thresholded_l1_loss + self.l2_weight * non_thresholded_l2_loss

        # Apply reduction (mean or sum) to each part
        if self.reduction == 'mean':
            combined_thresholded_loss = combined_thresholded_loss.mean()
            combined_non_thresholded_loss = combined_non_thresholded_loss.mean()
        elif self.reduction == 'sum':
            combined_thresholded_loss = combined_thresholded_loss.sum()
            combined_non_thresholded_loss = combined_non_thresholded_loss.sum()

        # Combine thresholded and non-thresholded losses with respective weights
        total_loss = self.threshold_weight * combined_thresholded_loss + self.non_threshold_weight * combined_non_thresholded_loss

        return total_loss


# Example usage
if __name__ == "__main__":
    # Create example images (generated and target) with random pixel values
    gen_img = torch.rand((1, 3, 256, 256))  # Generated image
    target_img = torch.rand((1, 3, 256, 256))  # Target image

    # Instantiate the flexible thresholded loss with normalization
    loss_fn = FlexibleThresholdedLoss(use_mse_threshold=True, use_mae_threshold=True, reduction='mean',
                                      l1_weight=0.5, l2_weight=0.5, threshold_weight=0.7, non_threshold_weight=0.3,
                                      mse_clip_ratio=2.0, mae_clip_ratio=1.5)

    # Calculate the loss
    loss = loss_fn(gen_img, target_img)
    print(f"Flexible Thresholded MSE & MAE Loss: {loss.item()}")
