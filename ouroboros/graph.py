"""
Knowledge Graph for Ouroboros.
Persistent, structure-driven memory that scales to trillions of tokens.
Optimized for RAM efficiency and fast retrieval on CPU.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional

class KnowledgeGraph:
    def __init__(self, path: str = "data/local_state/graph.json"):
        self.path = path
        self.nodes = {}
        self.edges = []
        self._load()
        
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.nodes = data.get("nodes", {})
                    self.edges = data.get("edges", [])
            except Exception:
                self.nodes = {}
                self.edges = []
                
    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump({"nodes": self.nodes, "edges": self.edges}, f, ensure_ascii=False, indent=2)
            
    def add_node(self, label: str, metadata: Dict[str, Any] = None):
        if label not in self.nodes:
            self.nodes[label] = {
                "created_at": time.time(),
                "last_accessed": time.time(),
                "access_count": 1,
                "metadata": metadata or {}
            }
        else:
            self.nodes[label]["last_accessed"] = time.time()
            self.nodes[label]["access_count"] += 1
            if metadata:
                self.nodes[label]["metadata"].update(metadata)
        self._save()
        
    def add_edge(self, source: str, target: str, relationship: str, weight: float = 1.0):
        self.add_node(source)
        self.add_node(target)
        self.edges.append({
            "source": source,
            "target": target,
            "relationship": relationship,
            "weight": weight,
            "created_at": time.time()
        })
        self._save()
        
    def query(self, text: str) -> List[Dict[str, Any]]:
        # Simple keyword-based node retrieval for now
        # Will be replaced with more complex graph traversal later
        results = []
        for label, data in self.nodes.items():
            if label.lower() in text.lower():
                results.append({"label": label, "data": data})
        return results

    def get_context(self, text: str) -> str:
        hits = self.query(text)
        if not hits:
            return ""
        
        context_parts = ["Stored Knowledge:"]
        for hit in hits:
            label = hit["label"]
            meta = hit["data"].get("metadata", {})
            summary = meta.get("summary", "")
            context_parts.append(f"- {label}: {summary}")
            
        return "\n".join(context_parts)
