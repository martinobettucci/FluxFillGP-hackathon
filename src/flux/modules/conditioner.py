from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            from mmgp import offload as offloadobj

            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, subfolder="tokenizer_2", max_length=max_length)
            
            path = version + "/text_encoder_2/T5Encoder.safetensors"
            #path = version + "/text_encoder_2/T5Encoder_quanto_int8.safetensors" # incumment this line to download a prequantized model instead

            self.hf_module: T5EncoderModel  = offloadobj.fast_load_transformers_model(path) 

            # self.hf_module: T5EncoderModel  = T5EncoderModel.from_pretrained(version, subfolder="text_encoder_2",  **hf_kwargs).to("cpu") 

            # offloadobj.save_model(self.hf_module, "T5Encoder.safetensors", config_path = "text_encoder_2_config.json")
    

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
