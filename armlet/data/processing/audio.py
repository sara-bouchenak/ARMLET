import os
import wave

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset


def validate_audio_metadata(
    subdata,
    data_key,
    num_classes: int | None,
):
    X, y = subdata

    assert "audio_path" in X.columns, (
        f"{data_key}: expected X to contain an 'audio_path' column."
    )
    assert "label" in y.columns, (
        f"{data_key}: expected y to contain a 'label' column."
    )
    assert len(X) == len(y), (
        f"{data_key}: X and y must have the same number of rows."
    )
    assert len(X) > 0, (
        f"{data_key}: cannot process an empty audio partition."
    )
    assert y.shape[1] == 1, (
        f"{data_key}: expected y to contain exactly one label column."
    )

    missing_paths = X.loc[
        ~X["audio_path"].map(os.path.isfile),
        "audio_path",
    ].head(3).tolist()
    assert not missing_paths, (
        f"{data_key}: audio files do not exist: {missing_paths}"
    )

    labels = torch.tensor(y["label"].values, dtype=torch.int64).flatten()
    if num_classes is not None:
        assert labels.min().item() >= 0
        assert labels.max().item() < num_classes, (
            f"{data_key}: labels must be in [0, {num_classes - 1}], "
            f"got max label {labels.max().item()}"
        )

    return X, y

def preprocess_audio_data(
    subdata,
    data_key,
    sample_rate: int = 16000,
    num_samples: int = 16000,
    normalization: str | None = "peak",
    allow_resample: bool = False,
    allow_multichannel_to_mono: bool = True,
):
    X, y = subdata

    waveforms = []
    for audio_path in X["audio_path"]:
        waveform = _load_and_fix_waveform(
            audio_path=audio_path,
            data_key=data_key,
            sample_rate=sample_rate,
            num_samples=num_samples,
            allow_resample=allow_resample,
            allow_multichannel_to_mono=allow_multichannel_to_mono,
        )
        waveform = _normalize_waveform(waveform, normalization)
        waveforms.append(waveform)

    X_tensor = torch.stack(waveforms)
    y_tensor = torch.tensor(y.values, dtype=torch.int64).flatten()

    assert X_tensor.shape == (len(X), 1, num_samples)
    assert X_tensor.dtype == torch.float32
    assert y_tensor.shape == (len(X),)
    assert y_tensor.dtype == torch.int64
    assert len(X_tensor) == len(y_tensor)

    return X_tensor, y_tensor

def _load_and_fix_waveform(
    audio_path,
    data_key,
    sample_rate,
    num_samples,
    allow_resample,
    allow_multichannel_to_mono,
):
    assert os.path.isfile(audio_path), (
        f"{data_key}: audio file does not exist: {audio_path}"
    )

    with wave.open(audio_path, "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        frames = wav_file.readframes(num_frames)

    assert sample_width == 2, (
        f"{data_key}: expected 16-bit PCM WAV, got sample width "
        f"{sample_width} bytes for {audio_path}"
    )

    waveform = torch.frombuffer(bytearray(frames), dtype=torch.int16)
    waveform = waveform.to(torch.float32).reshape(-1, num_channels).transpose(0, 1)
    waveform = waveform / 32768.0

    assert waveform.ndim == 2, (
        f"{data_key}: expected waveform [channels, samples], got {waveform.shape}"
    )

    if waveform.shape[0] > 1:
        assert allow_multichannel_to_mono, (
            f"{data_key}: multichannel audio found: {audio_path}"
        )
        waveform = waveform.mean(dim=0, keepdim=True)

    if source_sample_rate != sample_rate:
        assert allow_resample, (
            f"{data_key}: expected {sample_rate} Hz, "
            f"got {source_sample_rate} Hz for {audio_path}"
        )

        resampler = torchaudio.transforms.Resample(source_sample_rate, sample_rate, dtype=waveform.dtype)
        waveform = resampler(waveform)

    if waveform.shape[-1] > num_samples:
        waveform = waveform[:, :num_samples]
    elif waveform.shape[-1] < num_samples:
        waveform = F.pad(waveform, (0, num_samples - waveform.shape[-1]))

    assert waveform.shape == (1, num_samples), (
        f"{data_key}: expected [1, {num_samples}], got {waveform.shape}"
    )
    assert torch.isfinite(waveform).all(), (
        f"{data_key}: non-finite waveform values in {audio_path}"
    )

    return waveform.contiguous()

def _normalize_waveform(waveform, normalization):
    if normalization is None or normalization == "none":
        return waveform

    if normalization == "peak":
        peak = waveform.abs().max()
        if peak.item() > 0:
            return waveform / peak
        return waveform

    raise ValueError(f"Unknown normalization mode: {normalization}")

def make_lazy_audio_loader(
    subdata,
    cfg_loader_format,
    batch_size: int,
    shuffle: bool,
    data_key: str,
):

    X, y = subdata
    dataset = LazyAudioDataset(
        X=X,
        y=y,
        sample_rate=cfg_loader_format.sample_rate,
        num_samples=cfg_loader_format.num_samples,
        normalization=cfg_loader_format.normalization,
        allow_resample=cfg_loader_format.allow_resample,
        allow_multichannel_to_mono=cfg_loader_format.allow_multichannel_to_mono,
        data_key=data_key,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lazy_audio_collate_fn,
        num_workers=cfg_loader_format.num_workers,
        pin_memory=cfg_loader_format.pin_memory,
    )


class LazyAudioDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        data_key: str,
        sample_rate: int = 16000,
        num_samples: int = 16000,
        normalization: str | None = "peak",
        allow_resample: bool = False,
        allow_multichannel_to_mono: bool = True,
    ):

        self.audio_paths = X["audio_path"].tolist()
        self.labels = y["label"].astype("int64").tolist()
        self.data_key = data_key
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.normalization = normalization
        self.allow_resample = allow_resample
        self.allow_multichannel_to_mono = allow_multichannel_to_mono

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        label = self.labels[index]

        waveform = _load_and_fix_waveform(
            audio_path=audio_path,
            data_key=f"{self.data_key}[{index}]",
            sample_rate=self.sample_rate,
            num_samples=self.num_samples,
            allow_resample=self.allow_resample,
            allow_multichannel_to_mono=self.allow_multichannel_to_mono,
        )
        waveform = _normalize_waveform(waveform, self.normalization)

        return waveform, int(label)

def lazy_audio_collate_fn(batch):
    waveforms, labels = zip(*batch)
    X_batch = torch.stack(waveforms)
    y_batch = torch.tensor(labels, dtype=torch.int64)
    return X_batch, y_batch
