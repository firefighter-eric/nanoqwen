from __future__ import annotations

import torch

from nanoqwen.dpo import PreferenceDataset, collate_preference_batch, dpo_loss
from nanoqwen.sft import IGNORE_INDEX
from nanoqwen.tokenizer import ByteTokenizer


def supervised_text(labels, tokenizer: ByteTokenizer) -> str:
    ids = [int(token) for token in labels.tolist() if int(token) != IGNORE_INDEX]
    return tokenizer.decode(ids)


def test_preference_dataset_masks_prompt_tokens(tmp_path) -> None:
    tokenizer = ByteTokenizer()
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        '{"prompt":"user: greet\\nassistant: ","chosen":"hello","rejected":"goodbye"}\n',
        encoding="utf-8",
    )
    dataset = PreferenceDataset.from_jsonl(
        path=path,
        block_size=64,
        tokenizer=tokenizer,
    )
    chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = dataset[0]

    assert len(chosen_input_ids) == len(chosen_labels)
    assert len(rejected_input_ids) == len(rejected_labels)
    assert supervised_text(torch.tensor(chosen_labels), tokenizer).startswith("hello")
    assert "user:" not in supervised_text(torch.tensor(chosen_labels), tokenizer)
    assert supervised_text(torch.tensor(rejected_labels), tokenizer).startswith("goodbye")


def test_preference_collate_pads_labels_with_ignore_index(tmp_path) -> None:
    tokenizer = ByteTokenizer()
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        '{"prompt":"q: ","chosen":"a","rejected":"longer"}\n'
        '{"prompt":"q: ","chosen":"short","rejected":"b"}\n',
        encoding="utf-8",
    )
    dataset = PreferenceDataset.from_jsonl(
        path=path,
        block_size=32,
        tokenizer=tokenizer,
    )
    batch = collate_preference_batch([dataset[0], dataset[1]], pad_token_id=tokenizer.eos_token_id)

    assert batch.chosen_input_ids.shape[0] == 2
    assert batch.rejected_input_ids.shape[0] == 2
    assert (batch.chosen_labels == IGNORE_INDEX).any()
    assert (batch.rejected_labels == IGNORE_INDEX).any()


def test_dpo_loss_rewards_preferred_policy_direction() -> None:
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    good = dpo_loss(
        policy_chosen_logps=torch.tensor([2.0]),
        policy_rejected_logps=torch.tensor([0.0]),
        ref_chosen_logps=ref_chosen,
        ref_rejected_logps=ref_rejected,
        beta=1.0,
    )
    bad = dpo_loss(
        policy_chosen_logps=torch.tensor([0.0]),
        policy_rejected_logps=torch.tensor([2.0]),
        ref_chosen_logps=ref_chosen,
        ref_rejected_logps=ref_rejected,
        beta=1.0,
    )

    assert good.loss < bad.loss
    assert good.accuracy.item() == 1.0
    assert bad.accuracy.item() == 0.0

