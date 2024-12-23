import random
from tqdm import tqdm
from pathlib import Path
from enum import Enum
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .module import AutoEncoder, PDN_S, PDN_M


class EfficientADModelSize(str, Enum):
    """Supported EfficientAD model sizes"""

    M = "medium"
    S = "small"


class EfficientAD(nn.Module):
    def __init__(
        self,
        teacher_out_channels: int,
        input_size: tuple[int, int],
        model_size: EfficientADModelSize = EfficientADModelSize.S,
        padding=False,
        pad_maps=True,
    ):
        super(EfficientAD, self).__init__()
        self.pad_maps = pad_maps
        self.teacher: PDN_M | PDN_S
        self.student: PDN_M | PDN_S
        if model_size == EfficientADModelSize.M:
            self.teacher = PDN_M(
                out_channels=teacher_out_channels, padding=padding).eval()
            self.student = PDN_M(
                out_channels=teacher_out_channels * 2, padding=padding)
        elif model_size == EfficientADModelSize.S:
            self.teacher = PDN_S(
                out_channels=teacher_out_channels, padding=padding).eval()
            self.student = PDN_S(
                out_channels=teacher_out_channels * 2, padding=padding)
        else:
            raise ValueError(f"Unknown model size {model_size}")

        pretrained_models_dir = Path(
            "task/anomaly/unsupervised/models/efficientad/teacher_weights/")
        teacher_path = (
            pretrained_models_dir / f"pretrained_teacher_{model_size}.pth"
        )
        self.teacher.load_state_dict(
            torch.load(teacher_path, map_location='cpu'))

        for parameters in self.teacher.parameters():
            parameters.requires_grad = False

        self.ae: AutoEncoder = AutoEncoder(
            out_channels=teacher_out_channels, padding=padding, img_size=input_size)
        self.teacher_out_channels: int = teacher_out_channels
        self.input_size: tuple[int, int] = input_size

        self.mean_std: nn.ParameterDict = nn.ParameterDict(
            {
                "mean": torch.zeros((1, self.teacher_out_channels, 1, 1)),
                "std": torch.zeros((1, self.teacher_out_channels, 1, 1)),
            }
        )

        self.quantiles: nn.ParameterDict = nn.ParameterDict(
            {
                "qa_st": torch.tensor(0.0),
                "qb_st": torch.tensor(0.0),
                "qa_ae": torch.tensor(0.0),
                "qb_ae": torch.tensor(0.0),
            }
        )

    def is_set(self, p_dic: nn.ParameterDict) -> bool:
        for _, value in p_dic.items():
            if value.sum() != 0:
                return True
        return False

    def choose_random_aug_image(self, image: Tensor) -> Tensor:
        transform_functions = [
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
        ]
        # Sample an augmentation coefficient λ from the uniform distribution U(0.8, 1.2)
        coefficient = random.uniform(0.8, 1.2)
        transform_function = random.choice(transform_functions)
        return transform_function(image, coefficient)

    @torch.no_grad()
    def teacher_channel_mean_std(self, dataloader: DataLoader) -> dict[str, Tensor]:
        """Calculate the mean and std of the teacher models activations.

        Args:
            dataloader (DataLoader): Dataloader of the respective dataset.

        Returns:
            dict[str, Tensor]: Dictionary of channel-wise mean and std
        """
        device = next(self.teacher.parameters()).device
        arrays_defined = False
        n: torch.Tensor | None = None
        chanel_sum: torch.Tensor | None = None
        chanel_sum_sqr: torch.Tensor | None = None

        for _, (x, _) in enumerate(tqdm(dataloader)):
            y = self.teacher(x.to(device, dtype=torch.float32))
            if not arrays_defined:
                _, num_channels, _, _ = y.shape
                n = torch.zeros((num_channels,),
                                dtype=torch.int64, device=y.device)
                chanel_sum = torch.zeros(
                    (num_channels,), dtype=torch.float64, device=y.device)
                chanel_sum_sqr = torch.zeros(
                    (num_channels,), dtype=torch.float64, device=y.device)
                arrays_defined = True

            n += y[:, 0].numel()
            chanel_sum += torch.sum(y, dim=[0, 2, 3])
            chanel_sum_sqr += torch.sum(y**2, dim=[0, 2, 3])

        channel_mean = chanel_sum / n

        channel_std = (torch.sqrt((chanel_sum_sqr / n) -
                       (channel_mean**2))).float()[None, :, None, None]
        channel_mean = channel_mean.float()[None, :, None, None]

        return {"mean": channel_mean, "std": channel_std}

    def forward(self, batch: Tensor, batch_imagenet: Tensor = None, normalize: bool = True):
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                teacher_output = (
                    teacher_output - self.mean_std["mean"]) / self.mean_std["std"]

        student_output = self.student(batch)
        distance_st = torch.pow(
            teacher_output - student_output[:, : self.teacher_out_channels, :, :], 2)

        if self.training:
            # Student loss
            distance_st = self.reduce_tensor_elems(distance_st)
            d_hard = torch.quantile(distance_st, 0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            student_output_penalty = self.student(
                batch_imagenet)[:, : self.teacher_out_channels, :, :]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty

            # Autoencoder and Student AE Loss
            aug_img = self.choose_random_aug_image(batch)
            ae_output_aug = self.ae(aug_img)

            with torch.no_grad():
                teacher_output_aug = self.teacher(aug_img)
                if self.is_set(self.mean_std):
                    teacher_output_aug = (
                        teacher_output_aug - self.mean_std["mean"]) / self.mean_std["std"]

            student_output_ae_aug = self.student(
                aug_img)[:, self.teacher_out_channels:, :, :]

            distance_ae = torch.pow(teacher_output_aug - ae_output_aug, 2)
            distance_stae = torch.pow(ae_output_aug - student_output_ae_aug, 2)

            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)

            return loss_st, loss_ae, loss_stae

        else:
            with torch.no_grad():
                ae_output = self.ae(batch)

            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_stae = torch.mean(
                (ae_output - student_output[:, self.teacher_out_channels:]) ** 2, dim=1, keepdim=True
            )

            if self.pad_maps:
                map_st = F.pad(map_st, (4, 4, 4, 4))
                map_stae = F.pad(map_stae, (4, 4, 4, 4))
            map_st = F.interpolate(map_st, size=(
                self.input_size[0], self.input_size[1]), mode="bilinear")
            map_stae = F.interpolate(map_stae, size=(
                self.input_size[0], self.input_size[1]), mode="bilinear")

            if self.is_set(self.quantiles) and normalize:
                map_st = 0.1 * (map_st - self.quantiles["qa_st"]) / (
                    self.quantiles["qb_st"] - self.quantiles["qa_st"])
                map_stae = (
                    0.1 * (map_stae - self.quantiles["qa_ae"]) / (
                        self.quantiles["qb_ae"] - self.quantiles["qa_ae"])
                )

            map_combined = 0.5 * map_st + 0.5 * map_stae

            return map_combined[:, 0, ...], map_st, map_stae

    @torch.no_grad()
    def map_norm_quantiles(self, dataloader: DataLoader) -> dict[str, Tensor]:
        """Calculate 90% and 99.5% quantiles of the student(st) and autoencoder(ae).

        Args:
            dataloader (DataLoader): Dataloader of the respective dataset.

        Returns:
            dict[str, Tensor]: Dictionary of both the 90% and 99.5% quantiles
            of both the student and autoencoder feature maps.
        """
        device = next(self.teacher.parameters()).device
        maps_st = []
        maps_ae = []

        for _, (image, _) in enumerate(tqdm(dataloader)):
            image = image.to(device, dtype=torch.float32)
            _, map_st, map_ae = self(image, normalize=False)
            maps_st.append(map_st)
            maps_ae.append(map_ae)

        qa_st, qb_st = self._get_quantiles_of_maps(maps_st)
        qa_ae, qb_ae = self._get_quantiles_of_maps(maps_ae)
        return {"qa_st": qa_st, "qa_ae": qa_ae, "qb_st": qb_st, "qb_ae": qb_ae}

    def _get_quantiles_of_maps(self, maps: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Calculate 90% and 99.5% quantiles of the given anomaly maps.

        If the total number of elements in the given maps is larger than 16777216
        the returned quantiles are computed on a random subset of the given
        elements.

        Args:
            maps (list[Tensor]): List of anomaly maps.

        Returns:
            tuple[Tensor, Tensor]: Two scalars - the 90% and the 99.5% quantile.
        """
        maps_flat = self.reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=0.9).to(maps_flat.device)
        qb = torch.quantile(maps_flat, q=0.995).to(maps_flat.device)
        return qa, qb

    def reduce_tensor_elems(self, tensor: torch.Tensor, m=2**24) -> torch.Tensor:
        """Flattens n-dimensional tensors,  selects m elements from it
        and returns the selected elements as tensor. It is used to select
        at most 2**24 for torch.quantile operation, as it is the maximum
        supported number of elements.
        https://github.com/pytorch/pytorch/blob/b9f81a483a7879cd3709fd26bcec5f1ee33577e6/aten/src/ATen/native/Sorting.cpp#L291

        Args:
            tensor (torch.Tensor): input tensor from which elements are selected
            m (int): number of maximum tensor elements. Default: 2**24

        Returns:
                Tensor: reduced tensor
        """
        tensor = torch.flatten(tensor)
        if len(tensor) > m:
            # select a random subset with m elements.
            perm = torch.randperm(len(tensor), device=tensor.device)
            idx = perm[:m]
            tensor = tensor[idx]
        return tensor
