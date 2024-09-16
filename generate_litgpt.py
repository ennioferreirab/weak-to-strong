import torch
from typing import List, Optional
from litgpt.model import GPT
from litgpt.tokenizer import Tokenizer
import torch.nn.functional as F


class EmulatorGenerator:
    def __init__(
        self,
        model: GPT,
        tokenizer: Tokenizer,
    ):
        # Model and tokenizer
        self.tokenizer = tokenizer
        self.model = model
        self.max_seq_len = model.max_seq_length
        # Define pad_id; using 0 assuming it's suitable for your tokenizer
        self.pad_id = 0
        self.eos_id = tokenizer.eos_id
        self.device = next(model.parameters()).device  # Get the device from the model

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Generate text from prompts using litgpt's GPT model.
        Processes each prompt individually to match litgpt's expected input shapes.
        """
        outputs = []
        for prompt in prompts:
            prompt_tokens = self.tokenizer.encode(prompt, device=self.device)
            prompt_length = prompt_tokens.size(0)
            total_len = min(self.max_seq_len, max_gen_len + prompt_length)
            tokens = torch.full((total_len,), self.pad_id, device=self.device, dtype=torch.long)
            tokens[:prompt_length] = prompt_tokens
            input_pos = torch.arange(0, total_len, device=self.device, dtype=torch.long)

            # Initialize kv_cache
            self.model.set_kv_cache(batch_size=1)
            self.model.eval()

            for cur_pos in range(prompt_length, total_len):
                idx = tokens[:cur_pos].unsqueeze(0)  # Shape: [1, cur_pos]
                pos = input_pos[:cur_pos]            # Shape: [cur_pos]

                logits = self.model(idx, pos)        # Output shape: [1, cur_pos, vocab_size]
                logits = logits[0, -1, :]            # Shape: [vocab_size]

                next_token = self.sample_next(logits.unsqueeze(0), temperature, top_p, top_k)
                next_token = next_token.item()
                tokens[cur_pos] = next_token

                if next_token == self.eos_id:
                    break

            # Truncate at eos_id if present
            eos_positions = (tokens == self.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                tokens = tokens[: eos_positions[0]]
            else:
                tokens = tokens[: cur_pos + 1]

            output_text = self.tokenizer.decode(tokens)
            outputs.append(output_text)
        return outputs

    def sample_next(
        self,
        logits: torch.FloatTensor,  # Shape: [1, vocab_size]
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Sample the next token from the logits using temperature, top_p, and top_k sampling.
        """
        logits = logits / temperature

        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Apply top_p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            indices_to_remove = cumulative_probs > top_p
            # Shift indices to keep at least one token
            indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
            indices_to_remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(indices_to_remove, float('-inf'))
            logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_token

    @torch.no_grad()
    def generate_with_ref(
        self,
        ref_base_model: GPT,
        ref_finetune_model: GPT,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        beta: float = 1.0,
    ) -> List[str]:
        """
        Generate text from prompts using litgpt's GPT model with reference models.
        Processes each prompt individually to match litgpt's expected input shapes.
        """
        outputs = []
        for prompt in prompts:
            prompt_tokens = self.tokenizer.encode(prompt, device=self.device)
            prompt_length = prompt_tokens.size(0)
            total_len = min(self.max_seq_len, max_gen_len + prompt_length)
            tokens = torch.full((total_len,), self.pad_id, device=self.device, dtype=torch.long)
            tokens[:prompt_length] = prompt_tokens
            input_pos = torch.arange(0, total_len, device=self.device, dtype=torch.long)

            # Initialize kv_cache for all models
            self.model.set_kv_cache(batch_size=1)
            ref_base_model.set_kv_cache(batch_size=1)
            ref_finetune_model.set_kv_cache(batch_size=1)
            self.model.eval()
            ref_base_model.eval()
            ref_finetune_model.eval()

            for cur_pos in range(prompt_length, total_len):
                idx = tokens[:cur_pos].unsqueeze(0)  # Shape: [1, cur_pos]
                pos = input_pos[:cur_pos]            # Shape: [cur_pos]

                # Get logits from all models
                logits = self.model(idx, pos)[:, -1, :]  # Shape: [1, vocab_size]
                ref_base_logits = ref_base_model(idx, pos)[:, -1, :].to(self.device)
                ref_finetune_logits = ref_finetune_model(idx, pos)[:, -1, :].to(self.device)

                # Compute adjusted logits
                ori_lprobs = F.log_softmax(logits / temperature, dim=-1)
                ref_base_lprobs = F.log_softmax(ref_base_logits / temperature, dim=-1)
                ref_finetune_lprobs = F.log_softmax(ref_finetune_logits / temperature, dim=-1)

                new_lprobs = ori_lprobs + beta * (ref_finetune_lprobs - ref_base_lprobs)
                # Normalize
                probs = torch.exp(new_lprobs)
                probs = probs / probs.sum(dim=-1, keepdim=True)

                # Sample next token
                next_token = self.sample_next_with_ref(probs, temperature, top_p, top_k)
                next_token = next_token.item()
                tokens[cur_pos] = next_token

                if next_token == self.eos_id:
                    break

            # Truncate at eos_id if present
            eos_positions = (tokens == self.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                tokens = tokens[: eos_positions[0]]
            else:
                tokens = tokens[: cur_pos + 1]

            output_text = self.tokenizer.decode(tokens)
            outputs.append(output_text)
        return outputs

    def sample_next_with_ref(
        self,
        probs: torch.FloatTensor,  # Shape: [1, vocab_size]
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Sample the next token using adjusted probabilities, temperature, top_p, and top_k sampling.
        """
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        if top_k is not None:
            top_k = min(top_k, probs.size(-1))
            indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
            probs = probs.masked_fill(indices_to_remove, 0)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            indices_to_remove = cumulative_probs > top_p
            # Shift indices to include at least one token
            indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
            indices_to_remove[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(indices_to_remove, 0)
            probs = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs)

        probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normalize
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_token