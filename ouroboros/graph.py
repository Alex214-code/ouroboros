"""
Knowledge Graph for Ouroboros.
Persistent, structure-driven associative memory that scales efficiently on CPU.
Uses keyword indexing and graph traversal for context retrieval.
"""

import json
import os
import time
import logging
from typing import Dict, List, Any, Optional, Set

log = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self, path: str = "data/local_state/graph.json"):
        self.path = os.path.abspath(path)
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self._load()
        
    def _load(self):
        if os.path.exists(self.path):
            try:
                # Use utf-8-sig to automatically handle UTF-8 BOM if present
                with open(self.path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                    self.nodes = data.get("nodes", {})
                    self.edges = data.get("edges", [])
                log.info(f"Graph: Loaded {len(self.nodes)} nodes and {len(self.edges)} edges.")
            except Exception as e:
                log.warning(f"Graph: Failed to load from {self.path}: {e}")
                self.nodes = {}
                self.edges = []
                
    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump({"nodes": self.nodes, "edges": self.edges}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"Graph: Failed to save to {self.path}: {e}")
            
    def add_node(self, label: str, metadata: Dict[str, Any] = None):
        """Adds or updates a node in the graph."""
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
                # Merge metadata carefully
                if "metadata" not in self.nodes[label]:
                    self.nodes[label]["metadata"] = {}
                self.nodes[label]["metadata"].update(metadata)
        self._save()
        return label
        
    def add_edge(self, source: str, target: str, relationship: str, weight: float = 1.0):
        """Creates a directional relationship between nodes."""
        # Ensure nodes exist
        if source not in self.nodes: self.add_node(source)
        if target not in self.nodes: self.add_node(target)
        
        # Check if edge already exists to prevent duplicates
        for edge in self.edges:
            if edge["source"] == source and edge["target"] == target and edge["relationship"] == relationship:
                edge["weight"] = (edge.get("weight", 1.0) + weight) / 2
                return
                
        self.edges.append({
            "source": source,
            "target": target,
            "relationship": relationship,
            "weight": weight,
            "created_at": time.time()
        })
        self._save()
        
    def query(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Finds most relevant nodes based on text input and graph connectivity."""
        query_terms = set(text.lower().split())
        scored_nodes = []
        
        for label, data in self.nodes.items():
            score = 0
            label_lower = label.lower()
            
            # 1. Direct label match (highest priority)
            if label_lower in text.lower():
                score += 10
            
            # 2. Term overlap
            label_terms = set(label_lower.split())
            overlap = len(query_terms.intersection(label_terms))
            score += overlap * 2
            
            # 3. Access count and recency (slight boost)
            recency_boost = 1 / (max(1, (time.time() - data["last_accessed"]) / 3600))
            score += min(2, data["access_count"] * 0.1 + recency_boost)
            
            if score > 0:
                scored_nodes.append((label, score, data))
                
        # Sort by score descending
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        return [{"label": n[0], "data": n[2], "score": n[1]} for n in scored_nodes[:limit]]

    def get_context(self, text: str) -> str:
        """Retrieves structured context from the graph for LLM injection."""
        hits = self.query(text)
        if not hits:
            return ""
        
        relevant_labels = [h["label"] for h in hits]
        context_parts = ["### Associated Knowledge:"]
        
        for hit in hits:
            label = hit["label"]
            meta = hit["data"].get("metadata", {})
            
            # Primary node info
            node_info = f"- **{label}**"
            if "type" in meta: node_info += f" ({meta['type']})"
            
            # Detail from summary or other fields
            detail = meta.get("summary") or meta.get("description") or ""
            if detail:
                node_info += f": {detail}"
            
            context_parts.append(node_info)
            
            # Find neighbors (associative memory)
            neighbors = []
            for edge in self.edges:
                if edge["source"] == label and edge["target"] not in relevant_labels:
                    neighbors.append(f"{edge['relationship']} -> {edge['target']}")
                elif edge["target"] == label and edge["source"] not in relevant_labels:
                    neighbors.append(f"{edge['source']} -> {edge['relationship']}")
            
            if neighbors:
                context_parts.append(f"  *Relations: {', '.join(neighbors[:3])}*")
            
        return "\n".join(context_parts)
