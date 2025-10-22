
# agents.py
# -*- coding: utf-8 -*-
"""
- Memory == list[str] of abstracts (no events/tags).
- MemoryAgent exposes only: memorize(message) -> MemoryUpdate
- ResearchAgent uses explicit Integrate(search_results, temp_memory) -> temp_memory.
Prompts are placeholders.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import re


# =============================
# Core data models
# =============================

@dataclass
class MemoryState:
    """Long-term memory: only abstracts list."""
    abstracts: List[str] = field(default_factory=list)


@dataclass
class Page:
    page_id: Optional[str]  # Optional; can be set by external caller or left None
    header: str
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryUpdate:
    new_state: MemoryState
    new_page: Page
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchPlan:
    info_needs: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    tool_inputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    keyword_collection: List[str] = field(default_factory=list)
    vector_queries: List[str] = field(default_factory=list)
    page_id_list: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    tool: str
    inputs: Dict[str, Any]
    outputs: Any
    error: Optional[str] = None   # None means success; otherwise error info.


@dataclass
class Hit:
    page_id: Optional[str]          # For tool results without a page, keep None
    snippet: str
    source: str                     # "keyword" | "vector" | "page_id" | "tool:<name>"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResults:
    """Combined results returned by _search before integration."""
    hits: List[Hit] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class TempMemory:
    """
    Evolving temporary memory in research loop.
    This is exactly: Integrate(search_results, temp_memory) -> temp_memory
    """
    hits: List[Hit] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    keep_page_ids: List[str] = field(default_factory=list)


@dataclass
class ReflectionDecision:
    enough: bool
    new_request: Optional[str]
    keep_page_ids: List[str] = field(default_factory=list)


@dataclass
class ResearchOutput:
    integrated_memory: str
    raw_memory: Dict[str, Any]


# =============================
# Minimal interfaces / protocols
# =============================

class MemoryStore(Protocol):
    def load(self) -> MemoryState: ...
    def save(self, state: MemoryState) -> None: ...


class PageStore(Protocol):
    def add(self, page: Page) -> None: ...
    def get(self, page_id: str) -> Optional[Page]: ...
    def list_all(self) -> List[Page]: ...


class Retriever(Protocol):
    """Unified interface for keyword / vector / page-id retrievers."""
    name: str
    def build(self, pages: List[Page]) -> None: ...
    def search(self, query: str, top_k: int = 10) -> List[Hit]: ...


class Tool(Protocol):
    name: str
    def run(self, **kwargs) -> ToolResult: ...


class ToolRegistry(Protocol):
    def run_many(self, tool_inputs: Dict[str, Dict[str, Any]]) -> List[ToolResult]: ...


# =============================
# In-memory default stores (for quick start)
# =============================

class InMemoryMemoryStore:
    def __init__(self, init_state: Optional[MemoryState] = None) -> None:
        self._state = init_state or MemoryState()

    def load(self) -> MemoryState:
        return self._state

    def save(self, state: MemoryState) -> None:
        self._state = state


class InMemoryPageStore:
    """
    Simple append-only list store for Page.
    page_id is optional; .get() returns the most recent page with that id.
    """
    def __init__(self) -> None:
        self._pages: List[Page] = []

    def add(self, page: Page) -> None:
        self._pages.append(page)

    def get(self, page_id: str) -> Optional[Page]:
        # Find last matching page_id
        for p in reversed(self._pages):
            if p.page_id == page_id:
                return p
        return None

    def list_all(self) -> List[Page]:
        return list(self._pages)


# =============================
# MemoryAgent
# =============================

class MemoryAgent:
    """
    Public API:
      - memorize(message) -> MemoryUpdate
    Internal only:
      - _decorate(message, memory_state) -> (header, decorated_new_page)
    Note: memory_state contains ONLY abstracts (list[str]).
    """

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        page_store: PageStore | None = None,
        llm: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.page_store = page_store or InMemoryPageStore()
        self.llm = llm  # optional, can be None for MVP

    # ---- Public ----
    def memorize(self, message: str) -> MemoryUpdate:
        """
        Update long-term memory with a new message and persist a decorated page.
        Steps:
          1) Create/Refine abstract (LLM optional; heuristic fallback)
          2) Merge into MemoryState (append unique abstract)
          3) _decorate(...) => header, decorated_new_page
          4) Write Page into page_store  (page_id left None by default)
        """
        message = message.strip()
        state = self.memory_store.load()

        # (1) Build abstract
        abstract = self._make_abstract(message)

        # (2) Merge into memory (only abstracts)
        # simple uniqueness check
        if abstract and abstract not in state.abstracts:
            state.abstracts.append(abstract)
        # keep last 100
        if len(state.abstracts) > 100:
            state.abstracts = state.abstracts[-100:]

        # (3) Decorate
        header, decorated_new_page = self._decorate(message, state)

        # (4) Persist page (no page_id assigned here)
        page = Page(page_id=None, header=header, content=message, meta={"decorated": decorated_new_page})
        self.page_store.add(page)
        self.memory_store.save(state)

        return MemoryUpdate(new_state=state, new_page=page, debug={"decorated_page": decorated_new_page})

    # ---- Internal helpers ----
    def _decorate(self, message: str, memory_state: MemoryState) -> Tuple[str, str]:
        """
        Private. Build a header from the current memory state and compose: "header; new_page".
        Header uses the latest abstract only.
        """
        latest_abs = memory_state.abstracts[-1] if memory_state.abstracts else ""
        header = f"[ABSTRACT] {latest_abs}".strip()
        decorated_new_page = f"{header}; {message}"
        return header, decorated_new_page

    def _make_abstract(self, message: str) -> str:
        """
        Produce a concise abstract for the message.
        - PROMPT PLACEHOLDER: if LLM is provided, call it.
        - Heuristic fallback: first sentence or leading ~200 chars.
        """
        if self.llm:
            # ---- PROMPT PLACEHOLDER ----
            # Replace with an LLM prompt that returns a concise abstract (<= 3 sentences).
            pass

        parts = re.split(r'[。.!?]\s*', message, maxsplit=1)
        base = parts[0] if (parts and parts[0]) else message[:200]
        # simple whitespace normalization inline
        return " ".join(base.split())[:200]


# =============================
# ResearchAgent
# =============================

class ResearchAgent:
    """
    Public API:
      - research(request) -> ResearchOutput
    Internal steps:
      - _planning(request, memory_state) -> SearchPlan
      - _search(plan) -> SearchResults  (calls keyword/vector/page_id + tools)
      - _integrate(search_results, temp_memory) -> TempMemory
      - _reflection(request, memory_state, temp_memory) -> ReflectionDecision

    Note: memory_state should be MemoryState with ONLY abstracts list.
    """

    def __init__(
        self,
        page_store: PageStore,
        tool_registry: Optional[ToolRegistry] = None,
        retrievers: Optional[Dict[str, Retriever]] = None,
        llm: Optional[Callable[[str], str]] = None,
        max_iters: int = 3,
        memory_state: Optional[MemoryState] = None,
    ) -> None:
        self.page_store = page_store
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.llm = llm  # optional
        self.max_iters = max_iters
        self.memory_state = memory_state or MemoryState()

        # Build indices upfront (if retrievers are provided)
        pages = self.page_store.list_all()
        for r in self.retrievers.values():
            try:
                r.build(pages)
            except Exception:
                pass

    # ---- Public ----
    def research(self, request: str) -> ResearchOutput:
        temp = TempMemory()
        iterations: List[Dict[str, Any]] = []
        next_request = request

        for step in range(self.max_iters):
            plan = self._planning(next_request, self.memory_state)

            search_results = self._search(plan)

            temp = self._integrate(search_results, temp)

            decision = self._reflection(request, self.memory_state, temp)

            iterations.append({
                "step": step,
                "plan": plan.__dict__,
                "hits_snapshot": [h.__dict__ for h in temp.hits[-20:]],
                "decision": decision.__dict__,
            })

            if decision.enough:
                break

            if not decision.new_request:
                break

            next_request = decision.new_request
            # extend keep ids (unique while preserving order)
            for pid in decision.keep_page_ids:
                if pid not in temp.keep_page_ids:
                    temp.keep_page_ids.append(pid)

        integrated = self._summarize(temp, request)
        raw = {
            "iterations": iterations,
            "final_hits": [h.__dict__ for h in temp.hits],
            "notes": temp.notes,
            "keep_page_ids": temp.keep_page_ids,
        }
        return ResearchOutput(integrated_memory=integrated, raw_memory=raw)

    # ---- Internal ----
    def _planning(self, request: str, memory_state: MemoryState) -> SearchPlan:
        """
        Produce a SearchPlan:
          - what specific info is needed
          - which tools are useful + inputs
          - keyword/vector/page_id payloads
        PROMPT PLACEHOLDER: if LLM provided, call it; else heuristic.
        """
        if self.llm:
            # ---- PROMPT PLACEHOLDER ----
            # Compose a planning prompt using memory_state.abstracts as context.
            # Parse JSON to SearchPlan.
            pass

        # Heuristic: request tokens + last 3 abstracts tokens
        req_tokens = re.findall(r"[A-Za-z\u4e00-\u9fa5]+", request)
        mem_tokens: List[str] = []
        for a in memory_state.abstracts[-3:]:
            mem_tokens.extend(re.findall(r"[A-Za-z\u4e00-\u9fa5]+", a))

        keywords: List[str] = []
        for t in req_tokens + mem_tokens:
            t = t.lower()
            if t not in keywords:
                keywords.append(t)
            if len(keywords) >= 8:
                break

        info_needs = [f"Need: {k}" for k in keywords] or [f"Clarify: {request[:80]}"]
        return SearchPlan(
            info_needs=info_needs,
            tools=[],
            tool_inputs={},
            keyword_collection=keywords[:5],
            vector_queries=[request],
            page_id_list=[],
        )

    def _search(self, plan: SearchPlan) -> SearchResults:
        """
        Unified search:
          1) Tools
          2) Keyword retriever
          3) Vector retriever
          4) Page-id lookup
        Returns SearchResults; integration is done separately.
        """
        hits: List[Hit] = []
        notes: List[str] = []

        # 1) Tools
        if plan.tools and self.tools is not None:
            tool_results = self.tools.run_many(plan.tool_inputs)
            for tr in tool_results:
                if tr.error is None:
                    snippet = str(tr.outputs)[:280]
                    hits.append(Hit(page_id=None, snippet=snippet, source=f"tool:{tr.tool}", meta={"inputs": tr.inputs}))
                else:
                    notes.append(f"Tool {tr.tool} error: {tr.error}")

        # 2) Keyword
        for kw in plan.keyword_collection:
            hits.extend(self._search_by_keyword(kw, top_k=8))

        # 3) Vector
        for vq in plan.vector_queries:
            hits.extend(self._search_by_vector(vq, top_k=8))

        # 4) Page ID
        if plan.page_id_list:
            hits.extend(self._search_by_page_id(plan.page_id_list))

        return SearchResults(hits=hits, notes=notes)

    def _integrate(self, search_results: SearchResults, temp_memory: TempMemory) -> TempMemory:
        """
        Integrate(search_results, temp_memory) -> temp_memory
        - Deduplicate hits
        - Append notes
        - Maintain/extend keep_page_ids (include any page_ids from new hits)
        """
        merged_hits = self._merge_hits(temp_memory.hits, search_results.hits)

        # append notes (unique keep order)
        merged_notes = list(temp_memory.notes)
        for n in search_results.notes:
            if n not in merged_notes:
                merged_notes.append(n)

        keep_ids = list(temp_memory.keep_page_ids)
        for h in search_results.hits:
            if h.page_id and h.page_id not in keep_ids:
                keep_ids.append(h.page_id)

        return TempMemory(hits=merged_hits, notes=merged_notes, keep_page_ids=keep_ids)

    # ---- search channels ----
    def _search_by_keyword(self, query: str, top_k: int = 10) -> List[Hit]:
        r = self.retrievers.get("keyword")
        if r is not None:
            try:
                return r.search(query, top_k=top_k)
            except Exception:
                return []
        # naive fallback: scan pages for substring
        out: List[Hit] = []
        q = query.lower()
        for p in self.page_store.list_all():
            if q in p.content.lower() or q in p.header.lower():
                snippet = p.content[:200]
                out.append(Hit(page_id=p.page_id, snippet=snippet, source="keyword", meta={}))
                if len(out) >= top_k:
                    break
        return out

    def _search_by_vector(self, query: str, top_k: int = 10) -> List[Hit]:
        r = self.retrievers.get("vector")
        if r is not None:
            try:
                return r.search(query, top_k=top_k)
            except Exception:
                return []
        # fallback: none
        return []

    def _search_by_page_id(self, page_ids: List[str]) -> List[Hit]:
        out: List[Hit] = []
        for pid in page_ids:
            p = self.page_store.get(pid)
            if p:
                out.append(Hit(page_id=p.page_id, snippet=p.content[:200], source="page_id", meta={}))
        return out

    # ---- reflection & summarization ----
    def _reflection(self, request: str, memory_state: MemoryState, temp_memory: TempMemory) -> ReflectionDecision:
        """
        PROMPT PLACEHOLDER:
          If an LLM is available, judge sufficiency of information and generate a new_request if insufficient.
        Heuristic fallback:
          enough = >=5 hits OR overlap of request tokens with evidence >= 5.
        """
        if self.llm:
            # ---- PROMPT PLACEHOLDER ----
            # Craft a reflection prompt; context: memory_state.abstracts & temp_memory.hits
            pass

        req_tokens = set(re.findall(r"[A-Za-z\u4e00-\u9fa5]+", request.lower()))
        overlap = 0
        for h in temp_memory.hits:
            text = h.snippet.lower()
            for t in req_tokens:
                if t and t in text:
                    overlap += 1
        enough_by_overlap = overlap >= 5
        enough = enough_by_overlap or (len(temp_memory.hits) >= 5)

        new_request = None if enough else f"Add more specific details to cover unmet info needs about: {request[:100].strip()}"
        # keep top-5 page_ids by frequency
        freq: Dict[str, int] = {}
        for h in temp_memory.hits:
            if h.page_id:
                freq[h.page_id] = freq.get(h.page_id, 0) + 1
        keep_ids = [pid for pid, _cnt in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:5]]

        return ReflectionDecision(enough=enough, new_request=new_request, keep_page_ids=keep_ids)

    def _merge_hits(self, old_hits: List[Hit], new_hits: List[Hit]) -> List[Hit]:
        """Deduplicate by (page_id, normalized_snippet, source)."""
        seen: set[Tuple[Optional[str], str, str]] = set()
        out: List[Hit] = []

        def normalize_snippet(s: str) -> str:
            # collapse whitespace inline; limit length for dedup key
            return " ".join(s.split())[:160]

        def key(h: Hit) -> Tuple[Optional[str], str, str]:
            return (h.page_id, normalize_snippet(h.snippet), h.source)

        for h in old_hits + new_hits:
            k = key(h)
            if k not in seen:
                seen.add(k)
                out.append(h)
        # cap size to avoid unbounded growth
        return out[-200:]

    def _summarize(self, temp_memory: TempMemory, request: str) -> str:
        """
        Produce a concise integrated summary for the user, with light traceability.
        PROMPT PLACEHOLDER: replace heuristic with LLM summarization if available.
        """
        if self.llm:
            # ---- PROMPT PLACEHOLDER ----
            # Summarize hits + notes into a final answer.
            pass

        lines = [f"# Integrated Summary for: {request}",
                 f"- total_hits: {len(temp_memory.hits)}",
                 f"- kept_pages: {len(temp_memory.keep_page_ids)}"]
        if temp_memory.hits:
            lines.append("## Evidence preview")
            for h in temp_memory.hits[:5]:
                pid = h.page_id or "N/A"
                snippet = " ".join(h.snippet.split())[:140]
                lines.append(f"  • [{h.source}] pid={pid} :: {snippet}")
        if temp_memory.notes:
            lines.append("## Notes")
            for n in temp_memory.notes[-5:]:
                lines.append(f"- {n}")
        return "\n".join(lines)
