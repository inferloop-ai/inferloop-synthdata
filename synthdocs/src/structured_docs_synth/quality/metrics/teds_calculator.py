"""
TEDS (Tree Edit Distance based Similarity) calculator for table structure evaluation.
Measures table structure similarity using tree edit distance algorithms.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

from pydantic import BaseModel, Field

from ...core.config import BaseConfig
from ...core.exceptions import ValidationError
from ...core.logging import get_logger

logger = get_logger(__name__)


class TableNodeType(Enum):
    """Table structure node types"""
    TABLE = "table"
    THEAD = "thead"
    TBODY = "tbody"
    TR = "tr"
    TH = "th"
    TD = "td"
    EMPTY = "empty"


@dataclass
class TableNode:
    """Table structure tree node"""
    node_type: TableNodeType
    text: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List['TableNode'] = field(default_factory=list)
    row_index: Optional[int] = None
    col_index: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "type": self.node_type.value,
            "text": self.text,
            "attributes": self.attributes,
            "children": [child.to_dict() for child in self.children],
            "row_index": self.row_index,
            "col_index": self.col_index
        }
    
    def add_child(self, child: 'TableNode') -> None:
        """Add child node"""
        self.children.append(child)
    
    def get_text_content(self) -> str:
        """Get all text content from this node and children"""
        texts = []
        if self.text:
            texts.append(self.text.strip())
        
        for child in self.children:
            child_text = child.get_text_content()
            if child_text:
                texts.append(child_text)
        
        return " ".join(texts)
    
    def count_nodes(self) -> int:
        """Count total nodes in subtree"""
        count = 1
        for child in self.children:
            count += child.count_nodes()
        return count


@dataclass
class EditOperation:
    """Tree edit operation"""
    operation: str  # "insert", "delete", "substitute"
    node1: Optional[TableNode] = None
    node2: Optional[TableNode] = None
    cost: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "operation": self.operation,
            "node1_type": self.node1.node_type.value if self.node1 else None,
            "node2_type": self.node2.node_type.value if self.node2 else None,
            "cost": self.cost
        }


@dataclass
class TEDSResult:
    """TEDS calculation result"""
    teds_score: float
    edit_distance: int
    normalized_distance: float
    max_nodes: int
    operations: List[EditOperation] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "teds_score": self.teds_score,
            "edit_distance": self.edit_distance,
            "normalized_distance": self.normalized_distance,
            "max_nodes": self.max_nodes,
            "operations": [op.to_dict() for op in self.operations]
        }


class TEDSConfig(BaseConfig):
    """TEDS calculation configuration"""
    node_type_weight: float = Field(default=1.0, description="Weight for node type changes")
    text_weight: float = Field(default=0.5, description="Weight for text content changes")
    structure_weight: float = Field(default=2.0, description="Weight for structural changes")
    ignore_text: bool = Field(default=False, description="Ignore text content in comparison")
    normalize_text: bool = Field(default=True, description="Normalize text for comparison")
    case_sensitive: bool = Field(default=False, description="Case-sensitive text comparison")


class TEDSCalculator:
    """
    TEDS (Tree Edit Distance based Similarity) calculator.
    Evaluates table structure similarity using tree edit distance.
    """
    
    def __init__(self, config: Optional[TEDSConfig] = None):
        """Initialize TEDS calculator"""
        self.config = config or TEDSConfig()
        logger.info("Initialized TEDSCalculator")
    
    def calculate_teds(
        self,
        predicted_table: Union[TableNode, Dict[str, Any], str],
        ground_truth_table: Union[TableNode, Dict[str, Any], str]
    ) -> TEDSResult:
        """
        Calculate TEDS score between predicted and ground truth tables.
        
        Args:
            predicted_table: Predicted table structure
            ground_truth_table: Ground truth table structure
        
        Returns:
            TEDS calculation result
        """
        # Convert inputs to TableNode format
        pred_tree = self._parse_table_input(predicted_table)
        gt_tree = self._parse_table_input(ground_truth_table)
        
        # Calculate tree edit distance
        edit_distance, operations = self._calculate_tree_edit_distance(pred_tree, gt_tree)
        
        # Calculate TEDS score
        max_nodes = max(pred_tree.count_nodes(), gt_tree.count_nodes())
        normalized_distance = edit_distance / max_nodes if max_nodes > 0 else 0.0
        teds_score = 1.0 - normalized_distance
        
        return TEDSResult(
            teds_score=max(0.0, teds_score),
            edit_distance=edit_distance,
            normalized_distance=normalized_distance,
            max_nodes=max_nodes,
            operations=operations
        )
    
    def _parse_table_input(self, table_input: Union[TableNode, Dict[str, Any], str]) -> TableNode:
        """Parse various table input formats to TableNode"""
        if isinstance(table_input, TableNode):
            return table_input
        
        elif isinstance(table_input, dict):
            return self._dict_to_table_node(table_input)
        
        elif isinstance(table_input, str):
            # Try to parse as HTML or JSON
            if table_input.strip().startswith('<'):
                return self._html_to_table_node(table_input)
            else:
                try:
                    table_dict = json.loads(table_input)
                    return self._dict_to_table_node(table_dict)
                except json.JSONDecodeError:
                    # Treat as plain text table
                    return self._text_to_table_node(table_input)
        
        else:
            raise ValueError(f"Unsupported table input type: {type(table_input)}")
    
    def _dict_to_table_node(self, table_dict: Dict[str, Any]) -> TableNode:
        """Convert dictionary representation to TableNode"""
        if "type" not in table_dict:
            # Assume it's a table data structure
            return self._create_table_from_data(table_dict)
        
        node_type = TableNodeType(table_dict["type"])
        node = TableNode(
            node_type=node_type,
            text=table_dict.get("text"),
            attributes=table_dict.get("attributes", {}),
            row_index=table_dict.get("row_index"),
            col_index=table_dict.get("col_index")
        )
        
        # Recursively create children
        for child_dict in table_dict.get("children", []):
            child_node = self._dict_to_table_node(child_dict)
            node.add_child(child_node)
        
        return node
    
    def _create_table_from_data(self, table_data: Dict[str, Any]) -> TableNode:
        """Create TableNode from table data (rows/cells)"""
        table_node = TableNode(TableNodeType.TABLE)
        
        # Handle different data formats
        if "rows" in table_data:
            rows = table_data["rows"]
        elif "data" in table_data:
            rows = table_data["data"]
        else:
            # Assume the dict itself contains the table structure
            rows = []
            for key, value in table_data.items():
                if isinstance(value, list):
                    rows = value
                    break
        
        # Create table structure
        if rows:
            tbody = TableNode(TableNodeType.TBODY)
            table_node.add_child(tbody)
            
            for row_idx, row_data in enumerate(rows):
                tr_node = TableNode(TableNodeType.TR, row_index=row_idx)
                tbody.add_child(tr_node)
                
                if isinstance(row_data, list):
                    for col_idx, cell_data in enumerate(row_data):
                        cell_node = TableNode(
                            TableNodeType.TD,
                            text=str(cell_data) if cell_data is not None else "",
                            row_index=row_idx,
                            col_index=col_idx
                        )
                        tr_node.add_child(cell_node)
                elif isinstance(row_data, dict):
                    for col_idx, (key, value) in enumerate(row_data.items()):
                        cell_node = TableNode(
                            TableNodeType.TD,
                            text=str(value) if value is not None else "",
                            row_index=row_idx,
                            col_index=col_idx
                        )
                        tr_node.add_child(cell_node)
        
        return table_node
    
    def _html_to_table_node(self, html_string: str) -> TableNode:
        """Convert HTML table to TableNode (simplified parser)"""
        # Simple HTML table parser
        html_string = html_string.strip()
        
        # Remove HTML comments and extra whitespace
        html_string = re.sub(r'<!--.*?-->', '', html_string, flags=re.DOTALL)
        html_string = re.sub(r'\s+', ' ', html_string)
        
        # Create root table node
        table_node = TableNode(TableNodeType.TABLE)
        
        # Extract table content
        table_match = re.search(r'<table[^>]*>(.*?)</table>', html_string, re.DOTALL | re.IGNORECASE)
        if not table_match:
            return table_node
        
        table_content = table_match.group(1)
        
        # Parse tbody or direct tr elements
        tbody_match = re.search(r'<tbody[^>]*>(.*?)</tbody>', table_content, re.DOTALL | re.IGNORECASE)
        if tbody_match:
            tbody_node = TableNode(TableNodeType.TBODY)
            table_node.add_child(tbody_node)
            rows_content = tbody_match.group(1)
        else:
            tbody_node = table_node
            rows_content = table_content
        
        # Parse rows
        row_pattern = r'<tr[^>]*>(.*?)</tr>'
        row_matches = re.findall(row_pattern, rows_content, re.DOTALL | re.IGNORECASE)
        
        for row_idx, row_content in enumerate(row_matches):
            tr_node = TableNode(TableNodeType.TR, row_index=row_idx)
            tbody_node.add_child(tr_node)
            
            # Parse cells (td or th)
            cell_pattern = r'<(td|th)[^>]*>(.*?)</\1>'
            cell_matches = re.findall(cell_pattern, row_content, re.DOTALL | re.IGNORECASE)
            
            for col_idx, (cell_type, cell_content) in enumerate(cell_matches):
                node_type = TableNodeType.TH if cell_type.lower() == 'th' else TableNodeType.TD
                
                # Clean cell content
                cell_text = re.sub(r'<[^>]+>', '', cell_content).strip()
                
                cell_node = TableNode(
                    node_type,
                    text=cell_text,
                    row_index=row_idx,
                    col_index=col_idx
                )
                tr_node.add_child(cell_node)
        
        return table_node
    
    def _text_to_table_node(self, text_string: str) -> TableNode:
        """Convert plain text table to TableNode"""
        lines = text_string.strip().split('\n')
        table_node = TableNode(TableNodeType.TABLE)
        tbody_node = TableNode(TableNodeType.TBODY)
        table_node.add_child(tbody_node)
        
        for row_idx, line in enumerate(lines):
            if line.strip():
                tr_node = TableNode(TableNodeType.TR, row_index=row_idx)
                tbody_node.add_child(tr_node)
                
                # Split by common separators
                cells = re.split(r'[|\t,]', line)
                
                for col_idx, cell_text in enumerate(cells):
                    cell_node = TableNode(
                        TableNodeType.TD,
                        text=cell_text.strip(),
                        row_index=row_idx,
                        col_index=col_idx
                    )
                    tr_node.add_child(cell_node)
        
        return table_node
    
    def _calculate_tree_edit_distance(
        self,
        tree1: TableNode,
        tree2: TableNode
    ) -> Tuple[int, List[EditOperation]]:
        """Calculate tree edit distance between two table trees"""
        operations = []
        
        # If trees are identical
        if self._nodes_equal(tree1, tree2):
            return 0, operations
        
        # Base cases
        if tree1 is None and tree2 is None:
            return 0, operations
        
        if tree1 is None:
            cost = self._calculate_insert_cost(tree2)
            operations.append(EditOperation("insert", None, tree2, cost))
            return int(cost), operations
        
        if tree2 is None:
            cost = self._calculate_delete_cost(tree1)
            operations.append(EditOperation("delete", tree1, None, cost))
            return int(cost), operations
        
        # Calculate costs for different operations
        substitute_cost = 0
        substitute_ops = []
        
        if not self._nodes_equal(tree1, tree2):
            substitute_cost = self._calculate_substitute_cost(tree1, tree2)
            substitute_ops.append(EditOperation("substitute", tree1, tree2, substitute_cost))
        
        # Recursively calculate for children
        child_cost, child_ops = self._calculate_children_distance(tree1.children, tree2.children)
        
        total_cost = substitute_cost + child_cost
        all_operations = substitute_ops + child_ops
        
        return int(total_cost), all_operations
    
    def _calculate_children_distance(
        self,
        children1: List[TableNode],
        children2: List[TableNode]
    ) -> Tuple[int, List[EditOperation]]:
        """Calculate edit distance for children lists"""
        if not children1 and not children2:
            return 0, []
        
        if not children1:
            # Insert all children2
            total_cost = 0
            operations = []
            for child in children2:
                cost = self._calculate_insert_cost(child)
                total_cost += cost
                operations.append(EditOperation("insert", None, child, cost))
            return int(total_cost), operations
        
        if not children2:
            # Delete all children1
            total_cost = 0
            operations = []
            for child in children1:
                cost = self._calculate_delete_cost(child)
                total_cost += cost
                operations.append(EditOperation("delete", child, None, cost))
            return int(total_cost), operations
        
        # Use dynamic programming for sequence alignment
        m, n = len(children1), len(children2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + int(self._calculate_delete_cost(children1[i-1]))
        
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + int(self._calculate_insert_cost(children2[j-1]))
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                child1, child2 = children1[i-1], children2[j-1]
                
                # Calculate costs
                delete_cost = dp[i-1][j] + int(self._calculate_delete_cost(child1))
                insert_cost = dp[i][j-1] + int(self._calculate_insert_cost(child2))
                
                if self._nodes_equal(child1, child2):
                    substitute_cost = dp[i-1][j-1]
                else:
                    child_distance, _ = self._calculate_tree_edit_distance(child1, child2)
                    substitute_cost = dp[i-1][j-1] + child_distance
                
                dp[i][j] = min(delete_cost, insert_cost, substitute_cost)
        
        # Backtrack to get operations (simplified)
        operations = []
        return dp[m][n], operations
    
    def _nodes_equal(self, node1: TableNode, node2: TableNode) -> bool:
        """Check if two nodes are equal"""
        if node1.node_type != node2.node_type:
            return False
        
        if not self.config.ignore_text:
            text1 = self._normalize_text(node1.text or "")
            text2 = self._normalize_text(node2.text or "")
            if text1 != text2:
                return False
        
        return True
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not self.config.normalize_text:
            return text
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        if not self.config.case_sensitive:
            text = text.lower()
        
        return text
    
    def _calculate_substitute_cost(self, node1: TableNode, node2: TableNode) -> float:
        """Calculate cost of substituting node1 with node2"""
        cost = 0.0
        
        # Node type change cost
        if node1.node_type != node2.node_type:
            cost += self.config.node_type_weight
        
        # Text change cost
        if not self.config.ignore_text and node1.text != node2.text:
            cost += self.config.text_weight
        
        return cost
    
    def _calculate_insert_cost(self, node: TableNode) -> float:
        """Calculate cost of inserting a node"""
        cost = self.config.structure_weight
        
        if not self.config.ignore_text and node.text:
            cost += self.config.text_weight * 0.5  # Half weight for insertion
        
        return cost
    
    def _calculate_delete_cost(self, node: TableNode) -> float:
        """Calculate cost of deleting a node"""
        cost = self.config.structure_weight
        
        if not self.config.ignore_text and node.text:
            cost += self.config.text_weight * 0.5  # Half weight for deletion
        
        return cost
    
    def batch_calculate_teds(
        self,
        predicted_tables: List[Union[TableNode, Dict[str, Any], str]],
        ground_truth_tables: List[Union[TableNode, Dict[str, Any], str]]
    ) -> Dict[str, Any]:
        """Calculate TEDS for multiple table pairs"""
        if len(predicted_tables) != len(ground_truth_tables):
            raise ValueError("Number of predicted and ground truth tables must match")
        
        results = []
        teds_scores = []
        
        for i, (pred, gt) in enumerate(zip(predicted_tables, ground_truth_tables)):
            try:
                result = self.calculate_teds(pred, gt)
                results.append({
                    "index": i,
                    "result": result
                })
                teds_scores.append(result.teds_score)
            except Exception as e:
                logger.error(f"Failed to calculate TEDS for pair {i}: {str(e)}")
                results.append({
                    "index": i,
                    "error": str(e)
                })
        
        # Calculate aggregate statistics
        if teds_scores:
            aggregate_stats = {
                "total_pairs": len(predicted_tables),
                "successful_calculations": len(teds_scores),
                "failed_calculations": len(predicted_tables) - len(teds_scores),
                "mean_teds": statistics.mean(teds_scores),
                "median_teds": statistics.median(teds_scores),
                "min_teds": min(teds_scores),
                "max_teds": max(teds_scores),
                "std_teds": statistics.stdev(teds_scores) if len(teds_scores) > 1 else 0.0
            }
        else:
            aggregate_stats = {
                "total_pairs": len(predicted_tables),
                "successful_calculations": 0,
                "failed_calculations": len(predicted_tables),
                "error": "No successful TEDS calculations"
            }
        
        return {
            "results": results,
            "aggregate_statistics": aggregate_stats
        }
    
    def evaluate_table_structure_quality(
        self,
        table_structure: Union[TableNode, Dict[str, Any], str],
        reference_structures: Optional[List[Union[TableNode, Dict[str, Any], str]]] = None
    ) -> Dict[str, Any]:
        """Evaluate table structure quality"""
        table_node = self._parse_table_input(table_structure)
        
        # Basic structure analysis
        analysis = {
            "node_count": table_node.count_nodes(),
            "structure_depth": self._calculate_tree_depth(table_node),
            "row_count": self._count_rows(table_node),
            "column_count": self._count_columns(table_node),
            "cell_count": self._count_cells(table_node),
            "empty_cells": self._count_empty_cells(table_node),
            "structure_consistency": self._check_structure_consistency(table_node)
        }
        
        # Compare with references if provided
        if reference_structures:
            teds_scores = []
            for ref_structure in reference_structures:
                result = self.calculate_teds(table_structure, ref_structure)
                teds_scores.append(result.teds_score)
            
            analysis["reference_similarity"] = {
                "mean_teds": statistics.mean(teds_scores) if teds_scores else 0,
                "max_teds": max(teds_scores) if teds_scores else 0,
                "min_teds": min(teds_scores) if teds_scores else 0
            }
        
        return analysis
    
    def _calculate_tree_depth(self, node: TableNode) -> int:
        """Calculate maximum depth of tree"""
        if not node.children:
            return 1
        return 1 + max(self._calculate_tree_depth(child) for child in node.children)
    
    def _count_rows(self, table_node: TableNode) -> int:
        """Count number of rows in table"""
        count = 0
        
        def count_tr_nodes(node: TableNode):
            nonlocal count
            if node.node_type == TableNodeType.TR:
                count += 1
            for child in node.children:
                count_tr_nodes(child)
        
        count_tr_nodes(table_node)
        return count
    
    def _count_columns(self, table_node: TableNode) -> int:
        """Count maximum number of columns in table"""
        max_cols = 0
        
        def count_cols_in_row(tr_node: TableNode) -> int:
            return len([child for child in tr_node.children 
                       if child.node_type in [TableNodeType.TD, TableNodeType.TH]])
        
        def traverse(node: TableNode):
            nonlocal max_cols
            if node.node_type == TableNodeType.TR:
                cols = count_cols_in_row(node)
                max_cols = max(max_cols, cols)
            for child in node.children:
                traverse(child)
        
        traverse(table_node)
        return max_cols
    
    def _count_cells(self, table_node: TableNode) -> int:
        """Count total number of cells"""
        count = 0
        
        def count_cell_nodes(node: TableNode):
            nonlocal count
            if node.node_type in [TableNodeType.TD, TableNodeType.TH]:
                count += 1
            for child in node.children:
                count_cell_nodes(child)
        
        count_cell_nodes(table_node)
        return count
    
    def _count_empty_cells(self, table_node: TableNode) -> int:
        """Count number of empty cells"""
        count = 0
        
        def count_empty(node: TableNode):
            nonlocal count
            if node.node_type in [TableNodeType.TD, TableNodeType.TH]:
                text_content = node.get_text_content().strip()
                if not text_content:
                    count += 1
            for child in node.children:
                count_empty(child)
        
        count_empty(table_node)
        return count
    
    def _check_structure_consistency(self, table_node: TableNode) -> Dict[str, Any]:
        """Check table structure consistency"""
        rows = []
        
        def collect_rows(node: TableNode):
            if node.node_type == TableNodeType.TR:
                row_cells = [child for child in node.children 
                           if child.node_type in [TableNodeType.TD, TableNodeType.TH]]
                rows.append(len(row_cells))
            for child in node.children:
                collect_rows(child)
        
        collect_rows(table_node)
        
        if not rows:
            return {"consistent": True, "issues": []}
        
        # Check for consistent column count
        consistent = len(set(rows)) == 1
        issues = []
        
        if not consistent:
            issues.append(f"Inconsistent column count: {set(rows)}")
        
        return {
            "consistent": consistent,
            "column_counts": rows,
            "issues": issues
        }


def create_teds_calculator(
    config: Optional[Union[Dict[str, Any], TEDSConfig]] = None
) -> TEDSCalculator:
    """Factory function to create TEDS calculator"""
    if isinstance(config, dict):
        config = TEDSConfig(**config)
    return TEDSCalculator(config)


def calculate_teds_sample() -> TEDSResult:
    """Generate sample TEDS calculation for testing"""
    calculator = create_teds_calculator()
    
    # Sample table structures
    predicted = {
        "rows": [
            ["Name", "Age", "City"],
            ["John", "25", "NYC"],
            ["Jane", "30", "LA"]
        ]
    }
    
    ground_truth = {
        "rows": [
            ["Name", "Age", "Location"],
            ["John", "25", "New York"],
            ["Jane", "30", "Los Angeles"]
        ]
    }
    
    return calculator.calculate_teds(predicted, ground_truth)