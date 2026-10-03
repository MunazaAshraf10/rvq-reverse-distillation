"""Neural audio codecs at low bitrates, as reference points for the 2.1 kbps MiniMax code stream.

    encodec-2.2   EnCodec 32 kHz (the MusicGen tokenizer): 4 codebooks of 2048 at 50 Hz, 2.2 kbps
    dac-1.7       DAC 44.1 kHz: first 2 of 9 codebooks of 1024 at 86.13 Hz, 1.72 kbps
    dac-2.6       DAC 44.1 kHz: first 3 codebooks, 2.58 kbps

Both codecs are mono; they code the mono downmix at the stated bitrate, which is also what every
metric of this repository compares. Their goal (waveform reconstruction) differs from the MiniMax
code stream's (conditioning a generator), so they bound what a codec of that bitrate preserves
rather than compete on the same task.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import AutoProcessor, DacModel, EncodecModel

from rvq_ae.audio.io import resample


@dataclass(frozen=True, slots=True)
class CodecSpec:
    repo: str
    revision: str
    kind: str
    codebooks: int
    kbps: float


ENCODEC = ("facebook/encodec_32khz", "d0c45384f6c44db055f78200cfdcb9c1c8706727")
DAC = ("descript/dac_44khz", "c1bc521685adf9cfe247bc39a5ca58917eda1ac4")
CODECS = {
    "encodec-2.2": CodecSpec(*ENCODEC, "encodec", 4, 2.2),
    "dac-1.7": CodecSpec(*DAC, "dac", 2, 1.72),
    "dac-2.6": CodecSpec(*DAC, "dac", 3, 2.58),
}


class Codec:
    def __init__(self, name: str, device: torch.device) -> None:
        self.spec = CODECS[name]
        self.device = device
        model_class = EncodecModel if self.spec.kind == "encodec" else DacModel
        self.model = (
            model_class.from_pretrained(self.spec.repo, revision=self.spec.revision).to(device).eval()
        )
        self.processor = AutoProcessor.from_pretrained(self.spec.repo, revision=self.spec.revision)
        self.rate = int(self.processor.sampling_rate)

    @torch.no_grad()
    def __call__(self, audio: Tensor, rate: int) -> tuple[Tensor, int]:
        """Reconstruction [1, samples] of the mono downmix, at the codec's own rate."""
        wave = resample(audio.float().mean(0, keepdim=True), rate, self.rate)[0]
        inputs = self.processor(raw_audio=wave.numpy(), sampling_rate=self.rate, return_tensors="pt")
        inputs = inputs.to(self.device)
        if self.spec.kind == "encodec":
            bandwidth = self.spec.kbps
            encoded = self.model.encode(
                inputs["input_values"], inputs.get("padding_mask"), bandwidth=bandwidth
            )
            decoded = self.model.decode(encoded.audio_codes, encoded.audio_scales, inputs.get("padding_mask"))
            out = decoded.audio_values
        else:
            encoded = self.model.encode(inputs["input_values"], n_quantizers=self.spec.codebooks)
            out = self.model.decode(audio_codes=encoded.audio_codes).audio_values
        return out.reshape(1, -1)[:, : wave.shape[-1]].float().cpu(), self.rate
