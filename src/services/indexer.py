import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import zvec
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ..config import settings
from .file_filter import FileFilter
from .monitor import logger, monitor


# ── Text Chunker ────────────────────────────────────────────
class TextChunker:
    """Split text into overlapping chunks at sentence/word boundaries."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # Try to break at last newline or space within range
            if end < len(text):
                for sep in ("\n", ". ", " "):
                    last = text.rfind(sep, start + self.chunk_overlap, end)
                    if last > start:
                        end = last + len(sep)
                        break
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - self.chunk_overlap
        return chunks


# ── Constants ───────────────────────────────────────────────
BATCH_SIZE = 100       # max docs per zvec upsert
MAX_CHUNKS = 200       # max chunks per file (prevents huge files from exploding)
MANIFEST_NAME = ".source-mcp_manifest.json"


# ── Indexer Service ─────────────────────────────────────────
class IndexerService:
    def __init__(self):
        self.chunker = TextChunker()
        self.observer = Observer()
        self.collection = None
        self.file_filter: FileFilter | None = None
        self._manifest: Dict[str, dict] = {}  # {filepath: {mtime, size, chunks}}
        
        self.provider = None
        self.model_name = None
        self.fastembed_model = None
        self.openai_client = None
        self.reranker = None
        self._configured = False

    def configure(self):
        """Load settings and initialize components. Safe to call multiple times."""
        if self._configured:
            return

        self.provider = settings.embedding_provider
        self.model_name = settings.embedding_model
        
        if self.provider == "openai":
            import openai
            if not settings.openai_api_key:
                logger.warning("OpenAI provider selected but OPENAI_API_KEY not set. Falling back to FastEmbed.")
                self.provider = "fastembed"
            else:
                if not self.model_name:
                    self.model_name = "text-embedding-3-small"
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
                
        if self.provider == "fastembed":
            if not self.model_name:
                self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self.fastembed_model = TextEmbedding(model_name=self.model_name)
        
        # Initialize Reranker (Cross-Encoder)
        try:
            logger.info("Initializing Cross-Encoder for reranking...")
            # Use a lightweight but effective model (quantized)
            self.reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-Encoder initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Cross-Encoder: {e}")

        logger.info(f"Indexer configured: provider={self.provider}, model={self.model_name}")
        self._configured = True

    def initialize(self):
        """Open or create the Zvec collection. Must be called after settings are finalized."""
        if not self._configured:
            self.configure()

        if self.collection is not None:
            return self.collection
            
        db_path = Path(settings.zvec_path)
        
        # NOTE: Do NOT delete the LOCK file! Zvec uses OS-level flock() on it.
        # When the process exits (even crashes), the OS releases the flock automatically.
        # Deleting the LOCK file prevents zvec from creating/acquiring a new lock.

        if db_path.exists():
            try:
                contents = [p.name for p in db_path.iterdir()]
                logger.info(f"Checking existing DB at {db_path}. Contents: {contents}")
            except Exception as e:
                logger.warning(f"Failed to list DB directory: {e}")

        meta_path = db_path / "meta.json"
        
        current_dim = self._get_dimension()
        recreate = False
        
        # Check compatibility
        if db_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("dimension") != current_dim:
                    logger.warning(f"DB dimension mismatch ({meta.get('dimension')} vs {current_dim}). Recreating.")
                    recreate = True
                elif meta.get("provider") != self.provider:
                    # Optional: assume incompatible if provider changes, or just rely on dim
                    logger.info(f"Provider changed from {meta.get('provider')} to {self.provider}. Recreating for consistency.")
                    recreate = True
            except Exception as e:
                logger.warning(f"Error reading meta.json: {e}. Recreating DB.")
                recreate = True
        elif db_path.exists() and any(db_path.iterdir()):
             # Existing DB but no meta -> assume old fastembed 384
             if current_dim != 384:
                 logger.warning(f"Legacy DB found (assumed 384) but need {current_dim}. Recreating.")
                 recreate = True

        if recreate and db_path.exists():
            shutil.rmtree(db_path)
            # Clear manifest since DB is being recreated
            self._manifest = {}
            self._save_manifest()
            logger.info("Manifest cleared due to DB recreation (dimension/provider change).")
            
        # Ensure parent exists, but let zvec create the db dir itself
        if not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)

        if not db_path.exists(): 
             logger.info(f"Creating new Zvec collection at {settings.zvec_path} (dim={current_dim})")
             schema = zvec.CollectionSchema(
                name="knowledge_base",
                fields=[
                    zvec.FieldSchema(name="id", data_type=zvec.DataType.STRING),
                    zvec.FieldSchema(name="file_path", data_type=zvec.DataType.STRING),
                    zvec.FieldSchema(name="text", data_type=zvec.DataType.STRING),
                ],
                vectors=[
                    zvec.VectorSchema(
                        name="embedding",
                        dimension=current_dim,
                        data_type=zvec.DataType.VECTOR_FP32,
                    ),
                ],
            )
             self.collection = zvec.create_and_open(path=str(settings.zvec_path), schema=schema)
             # Save metadata
             meta_path.write_text(json.dumps({
                 "provider": self.provider,
                 "model": self.model_name,
                 "dimension": current_dim
             }))
             self.file_filter = FileFilter(Path(settings.docs_path))
             self._load_manifest()
             return self.collection

        logger.info(f"Opening existing Zvec collection at {settings.zvec_path}")
        try:
            self.collection = zvec.open(str(settings.zvec_path))
            self.file_filter = FileFilter(Path(settings.docs_path))
            self._load_manifest()
            return self.collection
        except Exception as e:
            logger.error(f"Failed to open existing DB: {e}. Recreating from scratch.")
            if db_path.exists():
                shutil.rmtree(db_path)
            # Clear manifest so files get re-indexed into the new empty DB
            self._manifest = {}
            self._save_manifest()
            logger.info("Manifest cleared due to DB recreation.")
            # Recursive call will set self.collection eventually
            return self.initialize()

    # ── Helpers & Internal ──────────────────────────────────
    def _get_dimension(self) -> int:
        if self.provider == "openai":
            if "3-small" in self.model_name:
                return 1536
            if "3-large" in self.model_name:
                return 3072
            if "ada-002" in self.model_name:
                return 1536
            return 1536 # Default fallback
        elif self.provider == "fastembed":
            # standard mini-lm is 384
            # We could try to access self.fastembed_model._model.get_sentence_embedding_dimension()
            return 384
        return 384

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        
        try:
            if self.provider == "openai":
                # OpenAI batch size limit? usually fine with small batches
                resp = self.openai_client.embeddings.create(
                    input=texts,
                    model=self.model_name
                )
                return [np.array(d.embedding, dtype=np.float32) for d in resp.data]
            
            elif self.provider == "fastembed":
                # generator
                return list(self.fastembed_model.embed(texts))
            
            return []
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

    def _file_fingerprint(self, path: Path) -> dict:
        try:
            stat = path.stat()
            return {"mtime": stat.st_mtime, "size": stat.st_size}
        except FileNotFoundError:
            return {}

    def _load_manifest(self):
        try:
            mp = Path(settings.docs_path) / MANIFEST_NAME
            if mp.exists():
                self._manifest = json.loads(mp.read_text())
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}")
            self._manifest = {}

    def _save_manifest(self):
        try:
            mp = Path(settings.docs_path) / MANIFEST_NAME
            mp.write_text(json.dumps(self._manifest, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save manifest: {e}")

    def _needs_reindex(self, path: Path) -> bool:
        path_str = str(path)
        if path_str not in self._manifest:
            return True
        entry = self._manifest[path_str]
        current_fp = self._file_fingerprint(path)
        old_fp = entry.get("fingerprint", {})
        return (
            current_fp.get("mtime") != old_fp.get("mtime") or 
            current_fp.get("size") != old_fp.get("size")
        )

    def _init_collection(self):
        """Helper to re-initialize collection (used by reindex)."""
        # Ensure configured
        self.configure()
        # Force re-creation logic reuse from initialize
        # But initialize checks if self.collection is not None
        # Caller of this should set self.collection = None
        return self.initialize()

    # ── Watching ────────────────────────────────────────────
    def start_watching(self):
        if self.observer.is_alive():
            return
        handler = DocsEventHandler(self)
        self.observer.schedule(handler, settings.docs_path, recursive=True)
        self.observer.start()
        logger.info(f"Started watching directory: {settings.docs_path}")

    def stop_watching(self):
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            # We need a new observer instance because they can't be restarted
            from watchdog.observers import Observer
            self.observer = Observer()
            logger.info("Stopped watching directory.")

    def reindex(self):
        """Wipe collection and manifest, then start full indexing in background."""
        logger.info("Forced reindex requested.")
        
        def run_reindex():
            try:
                self.stop_watching()
                
                # Clear manifest
                self._manifest = {}
                self._save_manifest()
                
                # Re-init collection - we'll force _init_collection to recreate by deleting first
                db_path = Path(settings.zvec_path)
                if db_path.exists():
                    try:
                        self.collection = None # Drop reference
                        shutil.rmtree(db_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete DB directory: {e}")

                self.collection = self._init_collection()
                self.file_filter = FileFilter(Path(settings.docs_path))
                self.index_directory()
                self.start_watching()
                logger.info("Forced reindex completed.")
            except Exception as e:
                logger.error(f"Error during reindex: {e}")

        import threading
        threading.Thread(target=run_reindex, daemon=True).start()

    # ── Full scan (incremental) ─────────────────────────────
    def index_directory(self):
        """Walk the docs directory — only index new/changed files."""
        indexable, skipped = self.file_filter.collect_files()

        # Split into new/changed vs unchanged
        to_index = [p for p in indexable if self._needs_reindex(p)]
        unchanged = len(indexable) - len(to_index)

        if unchanged > 0:
            logger.info(f"Skipping {unchanged} unchanged files (already indexed)")

        monitor.begin_scan(len(to_index), skipped=skipped)
        logger.info(
            f"Starting scan: {len(to_index)} to index, "
            f"{unchanged} unchanged, {skipped} filtered"
        )

        if not to_index:
            # Nothing to do — restore stats from manifest
            total_chunks = sum(m.get("chunks", 0) for m in self._manifest.values())
            size_mb = self._calc_index_size()
            monitor.update_stats(
                files_discovered=len(indexable),
                files_indexed=len(self._manifest),
                total_chunks=total_chunks,
            )
            monitor.finish_scan(index_size_mb=size_mb)
            logger.info(f"Index up to date. {len(self._manifest)} files, {total_chunks} chunks, {size_mb:.2f} MB")
            return

        for i, fpath in enumerate(to_index):
            self.index_file(str(fpath))
            if (i + 1) % 10 == 0:
                self._save_manifest()
                monitor.update_stats(index_size_mb=self._calc_index_size())

        self._save_manifest()
        size_mb = self._calc_index_size()
        monitor.finish_scan(index_size_mb=size_mb)
        logger.info(
            f"Finished scan. "
            f"Indexed {monitor.stats['files_indexed']}/{len(to_index)} new files, "
            f"{monitor.stats['total_chunks']} chunks, {size_mb:.2f} MB"
        )

    # ── Index a single file ─────────────────────────────────
    def index_file(self, file_path: str):
        try:
            path = Path(file_path)

            # Run through file filter (if available)
            if self.file_filter:
                reason = self.file_filter.should_index(path)
                if reason:
                    return

            if not path.exists() or not path.is_file():
                return

            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return

            chunks = self.chunker.split_text(text)
            if not chunks:
                return

            # Cap chunks per file
            if len(chunks) > MAX_CHUNKS:
                logger.warning(
                    f"Capping {path.name} from {len(chunks)} to {MAX_CHUNKS} chunks"
                )
                chunks = chunks[:MAX_CHUNKS]

            monitor.file_started(path.name)

            embeddings = self.embed(chunks)
            if not embeddings:
                monitor.file_failed()
                return

            # Build docs
            all_docs = []
            for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
                chunk_id = hashlib.md5(f"{file_path}:{i}".encode()).hexdigest()
                all_docs.append(zvec.Doc(
                    id=chunk_id,
                    fields={
                        "id": chunk_id,
                        "file_path": str(path),
                        "text": chunk,
                    },
                    vectors={
                        "embedding": vec,
                    },
                ))

            # Batch upsert to avoid "Too many docs" error
            for start in range(0, len(all_docs), BATCH_SIZE):
                batch = all_docs[start : start + BATCH_SIZE]
                self.collection.upsert(batch)

            monitor.file_indexed(len(chunks))
            # Record in manifest for incremental indexing
            self._manifest[str(path)] = {
                "fingerprint": self._file_fingerprint(path),
                "chunks": len(chunks),
            }
            logger.info(f"Indexed {path.name}: {len(chunks)} chunks.")

        except Exception as exc:
            logger.error(f"Error indexing {file_path}: {exc}")
            monitor.file_failed()

    # ── Query ───────────────────────────────────────────────
    def query(self, query_text: str, limit: int = 5, threshold: float = 0.0) -> List[str]:
        """
        Search with 3-stage Pipeline:
        1. Dense Retrieval (OpenAI/FastEmbed) -> 50 candidates
        2. Keyword Boosting (Sparse heuristic) -> 30 candidates
        3. Cross-Encoder Reranking (MsMarco) -> top K
        """
        try:
            vecs = self.embed([query_text])
            if not vecs:
                return []
            qvec = vecs[0]

            # 1. Fetch deep candidate pool (50 max)
            candidates_limit = min(limit * 10, 50)
            results = self.collection.query(
                vectors=[zvec.VectorQuery(field_name="embedding", vector=qvec)],
                topk=candidates_limit,
            )

            if not results:
                return []

            # 2. Keyword Boosting (Cheap Sparse)
            q_lower = query_text.lower().strip()
            q_tokens = [t for t in q_lower.split() if len(t) > 2]
            
            scored_candidates = []
            for res in results:
                if res.score is not None and res.score < threshold:
                    continue
                
                text = res.fields.get("text", "")
                text_lower = text.lower()
                
                # Base vector score
                score = res.score
                
                # Boosts
                if q_lower in text_lower:
                    score += 0.2
                
                matches = 0
                for token in q_tokens:
                    if token in text_lower:
                        matches += 1
                if matches > 0:
                    score += (matches * 0.03)

                scored_candidates.append({
                    "doc": res, 
                    "text": text, 
                    "initial_score": score
                })

            # Sort by boosted score and take top 30 for expensive reranking
            scored_candidates.sort(key=lambda x: x["initial_score"], reverse=True)
            rerank_candidates = scored_candidates[:30]

            # 3. Cross-Encoder Reranking (High Precision)
            # DISABLED: The default model is English-only and hurts Russian queries.
            # if self.reranker:
            #     docs_text = [c["text"] for c in rerank_candidates]
            #     try:
            #         # rank returns list of scores
            #         scores = list(self.reranker.rerank(query_text, docs_text))
            #         
            #         # Merge scores back
            #         for i, score in enumerate(scores):
            #             rerank_candidates[i]["final_score"] = score
            #         
            #         # Sort by Reranker score
            #         rerank_candidates.sort(key=lambda x: x["final_score"], reverse=True)
            #         
            #     except Exception as e:
            #         logger.warning(f"Reranking failed, falling back to initial scores: {e}")
            #         # Fallback: just use initial scores
            #         pass
            
            # ── Format Output ───────────────────────────────────────
            context: List[str] = []
            for item in rerank_candidates[:limit]:
                res = item["doc"]
                fpath = res.fields.get("file_path", "")
                fname = Path(fpath).name if fpath else "unknown"
                context.append(f"[{fname}] {item['text']}")
            
            return context

        except Exception as exc:
            logger.error(f"Query error: {exc}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    # ── Helpers ─────────────────────────────────────────────
    def _calc_index_size(self) -> float:
        try:
            total = sum(
                f.stat().st_size
                for f in Path(settings.zvec_path).rglob("*")
                if f.is_file()
            )
            return total / (1024 * 1024)
        except Exception:
            return 0.0

    def _get_total_vectors(self) -> int:
        try:
            # Check if collection is initialized
            if self.collection is None:
                return 0
            # zvec stats might be available
            if hasattr(self.collection, 'stats'):
                 return self.collection.stats.doc_count
            return 0
        except Exception:
            return 0

    def get_stats(self) -> Dict:
        return {
            "total_vectors": self._get_total_vectors(),
            "backend": "zvec",
        }


# ── File watcher ────────────────────────────────────────────
class DocsEventHandler(FileSystemEventHandler):
    def __init__(self, idx: IndexerService):
        self.indexer = idx

    def on_created(self, event):
        if not event.is_directory:
            self.indexer.index_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.indexer.index_file(event.src_path)


indexer = IndexerService()
