from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError
