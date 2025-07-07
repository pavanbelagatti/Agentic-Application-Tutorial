from importlib import metadata

from langchain_singlestore.cache import SingleStoreSemanticCache
from langchain_singlestore.chat_message_history import SingleStoreChatMessageHistory
from langchain_singlestore.document_loaders import SingleStoreLoader
from langchain_singlestore.vectorstores import SingleStoreVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "SingleStoreVectorStore",
    "SingleStoreSemanticCache",
    "SingleStoreChatMessageHistory",
    "SingleStoreLoader",
    "__version__",
]
