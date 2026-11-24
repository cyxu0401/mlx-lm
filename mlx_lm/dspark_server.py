# Copyright © 2026 MLX Contributors

from mlx_lm.models.cache import LRUPromptCache
from mlx_lm.server import APIHandler, ModelProvider, ResponseGenerator, _run_http_server
from mlx_lm.utils import _download

from . import dspark


class DSparkModelProvider(ModelProvider):
    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        super()._load(model_path, adapter_path)
        if draft_model_path is not None:
            self.draft_model = dspark.load_draft(
                _download(draft_model_path), self.model
            )
            self.model_key = (model_path, adapter_path, draft_model_path)
            self.is_batchable = False


class DSparkPromptCache(LRUPromptCache):
    def fetch_nearest_cache(self, model_key, tokens):
        if model_key[2] is not None:
            return None, tokens
        return super().fetch_nearest_cache(model_key, tokens)

    def insert_cache(self, model_key, tokens, prompt_cache, **kwargs):
        # DSpark must capture residuals for the full prompt on each request.
        if model_key[2] is None:
            super().insert_cache(model_key, tokens, prompt_cache, **kwargs)


class DSparkAPIHandler(APIHandler):
    def validate_model_parameters(self):
        super().validate_model_parameters()
        if self.requested_draft_model is not None:
            unsupported = (
                "min_p",
                "xtc_probability",
                "logit_bias",
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
            )
            if names := [name for name in unsupported if getattr(self, name)]:
                raise ValueError(f"DSpark does not support {', '.join(names)}.")


def serve(args):
    provider = DSparkModelProvider(args)
    if provider.is_distributed:
        raise ValueError("DSpark requires a single device.")
    generator = ResponseGenerator(provider, DSparkPromptCache(args.prompt_cache_size))
    try:
        _run_http_server(
            args.host, args.port, generator, handler_class=DSparkAPIHandler
        )
    finally:
        generator.stop_and_join()
