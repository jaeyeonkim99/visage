import math
import queue
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.models.bart.modeling_bart import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
    shift_tokens_right,
)
from transformers.utils import logging

from .visage_mono import (
    VisageConfig,
    VisageDecoder,
    VisageEncoder,
    VisageForConditionalGeneration,
    VisageModel,
    VisageOutput,
)

logger = logging.get_logger(__name__)


class VisageAmbisonicsEncoder(VisageEncoder):
    def __init__(
        self, config: VisageConfig, embed_tokens: Optional[nn.Embedding] = None
    ):
        super().__init__(config)
        self.direction_embedding = nn.Sequential(
            nn.Linear(3, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.energy_map_projection = nn.Sequential(
            nn.Linear(49, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        direction: Optional[torch.FloatTensor] = None,
        energy_map: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve inputs_embeds
        if inputs_embeds is None:
            raise ValueError("You have to specify inputs_embeds")
        bsz, seq, dim = inputs_embeds.shape

        # Prepare bos, eos
        bos_embeddings = torch.stack([self.bos] * bsz)
        eos_embeddings = torch.stack([self.eos] * bsz)

        # Prepare position ids
        position_ids = torch.arange(seq).to(self.device)

        # Prepare input embeddings
        inputs_embeds = self.clip_projection(inputs_embeds)
        if energy_map is not None:
            inputs_embeds = inputs_embeds + self.energy_map_projection(energy_map)

        embed_pos = self.embed_positions(position_ids)
        inputs_embeds = inputs_embeds + embed_pos

        if direction is not None:
            inputs_embeds = inputs_embeds + self.direction_embedding(
                direction
            ).unsqueeze(1)

        hidden_states = torch.cat(
            [bos_embeddings, inputs_embeds, eos_embeddings], dim=1
        )

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # expand attention_mask
        if attention_mask is not None:
            if self._use_flash_attention_2:
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self._use_sdpa and head_mask is None and not output_attentions:
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, inputs_embeds.dtype
                )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(
                    attention_mask, inputs_embeds.dtype
                )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class VisageAmbisonicsDecoder(VisageDecoder):
    def __init__(
        self, config: VisageConfig, embed_tokens: Optional[nn.Embedding] = None
    ):
        super().__init__(config)

        self.embed_dac = nn.ModuleList(
            [
                nn.Embedding(
                    math.ceil((config.dac_vocab_size + 1) / 64) * 64,
                    config.d_model,
                    padding_idx=config.dac_pad_token_id,
                )
                for _ in range(config.num_rvq * 4)
            ]
        )

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        direction: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        # Prepare inputs embeds
        bsz = input_ids.shape[0]

        # Prepare bos
        bos_embeddings = torch.stack([self.bos] * bsz)

        # Prepare inputs for first generation step
        if input_ids.ndim == 1:
            inputs_embeds = bos_embeddings
        else:
            seq_len = input_ids.shape[1]

            # Add dac embedding
            inputs_embeds = torch.zeros(bsz, seq_len, self.config.d_model).to(
                self.device
            )
            for i in range(self.num_rvq * 4):
                inputs_embeds = inputs_embeds + self.embed_dac[i](input_ids[..., i])

            # Do not concat bos during the generation using cache
            if seq_len != 1:
                inputs_embeds = torch.cat([bos_embeddings, inputs_embeds], dim=1)

        input_shape = inputs_embeds.size()[:-1]
        input = inputs_embeds[:, :, -1]

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._use_sdpa and not output_attentions and cross_attn_head_mask is None:
            # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self._use_flash_attention_2:
                encoder_attention_mask = (
                    encoder_attention_mask if 0 in encoder_attention_mask else None
                )
            elif (
                self._use_sdpa
                and cross_attn_head_mask is None
                and not output_attentions
            ):
                # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    (
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class VisageAmbisonicsModel(VisageModel):
    def __init__(self, config: VisageConfig):
        super().__init__(config)
        self.encoder = VisageAmbisonicsEncoder(config)
        self.decoder = VisageAmbisonicsDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        direction: Optional[torch.FloatTensor] = None,
        energy_map: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                direction=direction,
                energy_map=energy_map,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            direction=direction,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class VisageForAmbisonicsGeneration(VisageForConditionalGeneration):
    def __init__(self, config: VisageConfig):
        super().__init__(config)
        self.model = VisageAmbisonicsModel(config)

        self.dac_heads = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.dac_vocab_size)
                for _ in range(config.num_rvq * 4)
            ]
        )

        # Codebooks for each generation step
        self.even_heads = [0]
        self.odd_heads = []
        self.last_heads = []
        for idx in range(1, config.num_rvq * 4):
            if idx % config.num_rvq == 0 or idx < config.num_rvq:
                self.odd_heads.append(idx)
            else:
                self.even_heads.append(idx)
                self.last_heads.append(idx)
        self.guidance_debug = False

        self.post_init()

        if config.classifier_free_guidance:
            self.dir_null_embedding = nn.Parameter(torch.zeros(3))
            self.energy_null_embedding = nn.Parameter(torch.zeros(49))

    def init_spatial_from_pretrained(self):
        for idx in range(self.config.num_rvq, self.config.num_rvq * 4):
            copy_from = idx % self.config.num_rvq
            self.dac_heads[idx].load_state_dict(self.dac_heads[copy_from].state_dict())
            self.model.decoder.embed_dac[idx].load_state_dict(
                self.model.decoder.embed_dac[copy_from].state_dict()
            )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        direction: Optional[torch.FloatTensor] = None,
        energy_map: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if use_cache:
                logger.warning(
                    "The `use_cache` argument is changed to `False` since `labels` is provided."
                )
            use_cache = False

        if self.config.classifier_free_guidance:
            bsz, seq_len, dim = inputs_embeds.shape
            rnd = random.random()
            if rnd < self.config.cfg_dropout_prob:
                inputs_embeds = self.null_embedding.expand((bsz, seq_len, dim))
                energy_map = self.energy_null_embedding.expand((bsz, seq_len, 49))

            if direction is not None:
                rnd2 = random.random()
                if rnd2 < self.config.cfg_dropout_prob:
                    direction = self.dir_null_embedding.expand((bsz, 3))

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            direction=direction,
            energy_map=energy_map,
            **kwargs,
        )

        # Add logic for DAC logits and cross entropy loss
        generation_loss = None
        codebook_loss = []
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
            labels = labels.to(outputs[0].device)
            for i, dac_head in enumerate(self.dac_heads):
                dac_logit = dac_head(outputs.last_hidden_state)
                loss = loss_fct(
                    dac_logit.view(-1, self.config.dac_vocab_size),
                    labels[..., i].view(-1),
                )

                # Skip loss for batch with no z channels
                if not torch.isfinite(loss):
                    codebook_loss.append(0.0)
                    continue

                if generation_loss is None:
                    generation_loss = loss
                else:
                    generation_loss += loss
                codebook_loss.append(loss.item())

        if not return_dict:
            output = outputs[1:]
            return (
                ((generation_loss,) + output) if generation_loss is not None else output
            )

        return VisageOutput(
            loss=generation_loss,
            codebook_loss=codebook_loss,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @torch.no_grad()
    def generate_cfg(
        self,
        inputs_embeds,
        direction,
        energy_map,
        do_sample: bool = False,
        guidance_scale: float = -1.0,
        top_k: int = 256,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, dim = inputs_embeds.shape
        null_embeds = self.null_embedding.expand((bsz, seq_len, dim))
        null_direction = self.dir_null_embedding.expand((bsz, 3))
        null_energy_map = self.energy_null_embedding.expand((bsz, seq_len, 49))

        if guidance_scale < 0:
            guidance_scale = self.config.cfg_guidance_scale

        if self.guidance_debug:
            print(f"Both directional and visual guidance with scale: {guidance_scale}")
            self.guidance_debug = False

        # Prepare encoder output
        encoder_outputs = self.model.encoder(
            inputs_embeds=inputs_embeds,
            direction=direction,
            energy_map=energy_map,
            **kwargs,
        )
        uncond_encoder_outputs = self.model.encoder(
            inputs_embeds=null_embeds,
            direction=null_direction,
            energy_map=null_energy_map,
            **kwargs,
        )

        # Autoregresive generation
        target_len = int(
            inputs_embeds.shape[1]
            / self.config.clip_frame_rate
            * self.config.dac_frame_rate
        )
        generated = []
        logits = queue.Queue()
        uncond_logits = queue.Queue()
        for idx in range(target_len * 2 + 1):
            # Conditional generation
            if idx == 0:
                outputs = self.model.decoder(
                    input_ids=torch.zeros(bsz, dtype=torch.int64),
                    encoder_hidden_states=encoder_outputs[0],
                    use_cache=True,
                )
            else:
                outputs = self.model.decoder(
                    input_ids=next_decoder_input_ids,
                    encoder_hidden_states=encoder_outputs[0],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values

            # Unconditional generation
            if idx == 0:
                uncond_outputs = self.model.decoder(
                    input_ids=torch.zeros(bsz, dtype=torch.int64),
                    encoder_hidden_states=uncond_encoder_outputs[0],
                    use_cache=True,
                )
            else:
                uncond_outputs = self.model.decoder(
                    input_ids=next_decoder_input_ids,
                    encoder_hidden_states=uncond_encoder_outputs[0],
                    past_key_values=uncond_past_key_values,
                    use_cache=True,
                )
            uncond_past_key_values = uncond_outputs.past_key_values

            # Prepare dac logits with dac heads
            if idx == 0:
                head_list = [0]
            elif idx == target_len * 2:
                head_list = self.last_heads
            elif idx % 2 == 0:
                head_list = self.even_heads
            elif idx % 2 == 1:
                head_list = self.odd_heads

            for head_idx in head_list:
                logits.put(self.dac_heads[head_idx](outputs.last_hidden_state))
                uncond_logits.put(
                    self.dac_heads[head_idx](uncond_outputs.last_hidden_state)
                )

            next_decoder_input_ids = []
            for input_id_idx in range(self.config.num_rvq * 4):
                if input_id_idx in head_list:
                    # Classifier free guidance
                    logit = logits.get()
                    logit = logit[:, -1:]
                    uncond_logit = uncond_logits.get()
                    uncond_logit = uncond_logit[:, -1:]

                    logit = logit + (logit - uncond_logit) * guidance_scale

                    if do_sample:
                        logit = logit.squeeze(1)
                        indices_to_remove = (
                            logit < torch.topk(logit, top_k)[0][..., -1, None]
                        )
                        logit_processed = logit.masked_fill(
                            indices_to_remove, -float("Inf")
                        )
                        next_decoder_input_id = torch.multinomial(
                            F.softmax(logit_processed, dim=-1), num_samples=1
                        )
                    else:
                        next_decoder_input_id = logit.argmax(-1)
                    next_decoder_input_ids.append(next_decoder_input_id)
                else:
                    # Pad tokens
                    next_decoder_input_ids.append(
                        torch.full(
                            (bsz, 1), fill_value=self.config.dac_pad_token_id
                        ).to(inputs_embeds.device)
                    )

            # Concat and pad ids for each codebook
            next_decoder_input_ids = torch.stack(next_decoder_input_ids, dim=-1)

            # Update generated results
            generated.append(next_decoder_input_ids)

        # Unpad dac ids
        generated = torch.cat(generated, dim=1)
        unpadded = []
        for idx in range(self.config.num_rvq * 4):
            if idx == 0:
                unpadded.append(generated[:, :-1, idx][:, 0::2])
            elif idx < self.config.num_rvq or idx % self.config.num_rvq == 0:
                unpadded.append(generated[:, 1::2, idx])
            else:
                unpadded.append(generated[:, 2::2, idx])

        codes = torch.stack(unpadded, dim=-1)

        real_target_len = int(
            inputs_embeds.shape[1]
            / self.config.clip_frame_rate
            * self.config.dac_sample_rate
        )
        real_target_len = math.ceil(real_target_len / self.config.dac_downsample_rate)
        to_pad = real_target_len - codes.shape[1]
        if to_pad > 0:
            padding = codes[:, -1:].expand(-1, to_pad, -1)
            codes = torch.cat([codes, padding], dim=1)

        return codes

    @torch.no_grad()
    def generate_dual_cfg(
        self,
        inputs_embeds,
        direction,
        energy_map,
        guidance_scale: float = -1.0,
        spatial_guidance_scale: float = -1.0,
        do_sample: bool = False,
        top_k: int = 256,
        **kwargs,
    ) -> torch.Tensor:

        bsz, seq_len, dim = inputs_embeds.shape
        null_embeds = self.null_embedding.expand((bsz, seq_len, dim))
        null_direction = self.dir_null_embedding.expand((bsz, 3))
        null_energy_map = self.energy_null_embedding.expand((bsz, seq_len, 49))

        if guidance_scale < 0:
            guidance_scale = self.config.cfg_guidance_scale
        if spatial_guidance_scale < 0:
            spatial_guidance_scale = self.config.cfg_spatial_guidance_scale

        # Prepare encoder output
        encoder_outputs = self.model.encoder(
            inputs_embeds=inputs_embeds,
            energy_map=energy_map,
            direction=direction,
            **kwargs,
        )
        semantic_encoder_outputs = self.model.encoder(
            inputs_embeds=inputs_embeds,
            energy_map=energy_map,
            direction=null_direction,
            **kwargs,
        )
        spatial_encoder_outputs = self.model.encoder(
            inputs_embeds=null_embeds,
            energy_map=null_energy_map,
            direction=direction,
            **kwargs,
        )
        uncond_encoder_outputs = self.model.encoder(
            inputs_embeds=null_embeds,
            energy_map=null_energy_map,
            direction=null_direction,
            **kwargs,
        )

        target_len = int(
            inputs_embeds.shape[1]
            / self.config.clip_frame_rate
            * self.config.dac_frame_rate
        )
        generated = []
        logits = queue.Queue()
        uncond_logits = queue.Queue()
        semantic_logits = queue.Queue()
        spatial_logits = queue.Queue()

        for idx in range(target_len * 2 + 1):
            # Conditional generation: clip, energy, dir
            if idx == 0:
                outputs = self.model.decoder(
                    input_ids=torch.zeros(bsz, dtype=torch.int64),
                    encoder_hidden_states=encoder_outputs[0],
                    use_cache=True,
                )
            else:
                outputs = self.model.decoder(
                    input_ids=next_decoder_input_ids,
                    encoder_hidden_states=encoder_outputs[0],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values

            # Unconditional generation: null, null, null
            if idx == 0:
                uncond_outputs = self.model.decoder(
                    input_ids=torch.zeros(bsz, dtype=torch.int64),
                    encoder_hidden_states=uncond_encoder_outputs[0],
                    use_cache=True,
                )
            else:
                uncond_outputs = self.model.decoder(
                    input_ids=next_decoder_input_ids,
                    encoder_hidden_states=uncond_encoder_outputs[0],
                    past_key_values=uncond_past_key_values,
                    use_cache=True,
                )
            uncond_past_key_values = uncond_outputs.past_key_values

            # Semantic generation: clip, energy, null
            if idx == 0:
                semantic_uncond_outputs = self.model.decoder(
                    input_ids=torch.zeros(bsz, dtype=torch.int64),
                    encoder_hidden_states=semantic_encoder_outputs[0],
                    use_cache=True,
                )
            else:
                semantic_uncond_outputs = self.model.decoder(
                    input_ids=next_decoder_input_ids,
                    encoder_hidden_states=semantic_encoder_outputs[0],
                    past_key_values=semantic_uncond_past_key_values,
                    use_cache=True,
                )
            semantic_uncond_past_key_values = semantic_uncond_outputs.past_key_values

            # Spatial generation: null, null, direction
            if idx == 0:
                spatial_uncond_outputs = self.model.decoder(
                    input_ids=torch.zeros(bsz, dtype=torch.int64),
                    direction=direction,
                    encoder_hidden_states=spatial_encoder_outputs[0],
                    use_cache=True,
                )
            else:
                spatial_uncond_outputs = self.model.decoder(
                    input_ids=next_decoder_input_ids,
                    encoder_hidden_states=spatial_encoder_outputs[0],
                    past_key_values=spatial_uncond_past_key_values,
                    use_cache=True,
                )
            spatial_uncond_past_key_values = spatial_uncond_outputs.past_key_values

            # Prepare dac logits with dac heads
            if idx == 0:
                head_list = [0]
            elif idx == target_len * 2:
                head_list = self.last_heads
            elif idx % 2 == 0:
                head_list = self.even_heads
            elif idx % 2 == 1:
                head_list = self.odd_heads

            for head_idx in head_list:
                logits.put(self.dac_heads[head_idx](outputs.last_hidden_state))
                uncond_logits.put(
                    self.dac_heads[head_idx](uncond_outputs.last_hidden_state)
                )
                semantic_logits.put(
                    self.dac_heads[head_idx](semantic_uncond_outputs.last_hidden_state)
                )
                spatial_logits.put(
                    self.dac_heads[head_idx](spatial_uncond_outputs.last_hidden_state)
                )

            # Prepare Next decoder input ids based on logits
            next_decoder_input_ids = []
            for input_id_idx in range(self.config.num_rvq * 4):
                if input_id_idx in head_list:
                    # Classifier free guidance
                    logit = logits.get()
                    logit = logit[:, -1:]
                    uncond_logit = uncond_logits.get()
                    uncond_logit = uncond_logit[:, -1:]
                    spatial_logit = spatial_logits.get()
                    spatial_logit = spatial_logit[:, -1:]
                    semantic_logit = semantic_logits.get()
                    semantic_logit = semantic_logit[:, -1:]

                    logit = (
                        logit
                        + (semantic_logit - uncond_logit) * guidance_scale
                        + (spatial_logit - uncond_logit) * spatial_guidance_scale
                    )

                    if do_sample:
                        logit = logit.squeeze(1)
                        indices_to_remove = (
                            logit < torch.topk(logit, top_k)[0][..., -1, None]
                        )
                        logit_processed = logit.masked_fill(
                            indices_to_remove, -float("Inf")
                        )
                        next_decoder_input_id = torch.multinomial(
                            F.softmax(logit_processed, dim=-1), num_samples=1
                        )
                    else:
                        next_decoder_input_id = logit.argmax(-1)
                    next_decoder_input_ids.append(next_decoder_input_id)
                else:
                    # Pad tokens
                    next_decoder_input_ids.append(
                        torch.full(
                            (bsz, 1), fill_value=self.config.dac_pad_token_id
                        ).to(inputs_embeds.device)
                    )

            # Concat and pad ids for each codebook
            next_decoder_input_ids = torch.stack(next_decoder_input_ids, dim=-1)

            # Update generated results
            generated.append(next_decoder_input_ids)

        # Unpad dac ids
        generated = torch.cat(generated, dim=1)
        unpadded = []
        for idx in range(self.config.num_rvq * 4):
            if idx == 0:
                unpadded.append(generated[:, :-1, idx][:, 0::2])
            elif idx < self.config.num_rvq or idx % self.config.num_rvq == 0:
                unpadded.append(generated[:, 1::2, idx])
            else:
                unpadded.append(generated[:, 2::2, idx])

        codes = torch.stack(unpadded, dim=-1)

        real_target_len = int(
            inputs_embeds.shape[1]
            / self.config.clip_frame_rate
            * self.config.dac_sample_rate
        )
        real_target_len = math.ceil(real_target_len / self.config.dac_downsample_rate)
        to_pad = real_target_len - codes.shape[1]
        if to_pad > 0:
            padding = codes[:, -1:].expand(-1, to_pad, -1)
            codes = torch.cat([codes, padding], dim=1)

        return codes
