"""Dependency injection container for the OCR proxy application."""

from dependency_injector import containers, providers

from bentoml_ocr.ocr_proxy.backend import DefaultOCRBackend
from bentoml_ocr.ocr_proxy.config import AppConfig


class Container(containers.DeclarativeContainer):
    """DI container wiring configuration and the OCR backend provider."""

    config = providers.Singleton(AppConfig.from_env)
    backend = providers.Singleton(DefaultOCRBackend, config=config)
