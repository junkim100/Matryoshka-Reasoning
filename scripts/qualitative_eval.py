#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Weave Evaluation Pipeline for Matryoshka Reasoning

Enhanced evaluation script that provides comprehensive benchmarking capabilities
while maintaining backward compatibility with the original qualitative evaluation.

Features:
- MatryoshkaModel class with weave.Model integration
- Structured datasets organized by difficulty levels
- Custom scoring functions for reasoning quality, efficiency, and accuracy
- Multi-depth evaluation across different reasoning budgets
- Comprehensive benchmarking with statistical analysis
- Backward compatibility with original CLI interface

Usage:
  # Enhanced evaluation with structured datasets
  python scripts/qualitative_eval.py \
    --model output/my-matryoshka-8b \
    --project matryoshka-reasoning \
    --enhanced_eval True \
    --max_new_tokens 512

  # Original qualitative evaluation (backward compatible)
  python scripts/qualitative_eval.py \
    --model output/my-matryoshka-8b \
    --project matryoshka-reasoning \
    --max_new_tokens 512

  # Multi-depth benchmarking
  python scripts/qualitative_eval.py \
    --model output/my-matryoshka-8b \
    --project matryoshka-reasoning \
    --multi_depth_eval True \
    --budget_list "0,64,160,384,-1"

  # Comprehensive benchmarking with both enhanced and multi-depth evaluation
  python scripts/qualitative_eval.py \
    --model output/my-matryoshka-8b \
    --project matryoshka-reasoning \
    --enhanced_eval True \
    --multi_depth_eval True \
    --budget_list "0,64,160,384,-1"

New Features:
- MatryoshkaModel: Weave-compatible model wrapper with automatic versioning
- Structured Datasets: Organized by difficulty (easy/medium/hard) with metadata
- Custom Scorers: Reasoning quality, token efficiency, budget compliance, accuracy
- Multi-depth Evaluation: Systematic testing across different reasoning budgets
- Comprehensive Benchmarking: Statistical analysis and performance comparison
- Enhanced Logging: Detailed Weave integration with structured evaluation results
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
from datetime import datetime
from typing import List, Dict, Optional, Any

import fire
import weave

from matryoshka_infer import MatryoshkaEngine

# Global flag to track if Weave is available
WEAVE_AVAILABLE = False

# Default reasoning budgets for multi-depth evaluation
DEFAULT_BUDGETS = [0, 64, 160, 384, -1]

# Enhanced structured datasets organized by difficulty
STRUCTURED_DATASETS = {
    "easy": [
        {
            "id": "easy_1",
            "prompt": "What is 1+1?",
            "expected_answer": "2",
            "difficulty": "easy",
            "category": "arithmetic",
            "max_expected_budget": 32,
            "expected_reasoning_depth": "minimal",
        },
        {
            "id": "easy_2",
            "prompt": "If I have 5 apples and eat 2, how many do I have left?",
            "expected_answer": "3",
            "difficulty": "easy",
            "category": "arithmetic",
            "max_expected_budget": 32,
            "expected_reasoning_depth": "minimal",
        },
        {
            "id": "easy_3",
            "prompt": "What is 10 × 3?",
            "expected_answer": "30",
            "difficulty": "easy",
            "category": "arithmetic",
            "max_expected_budget": 32,
            "expected_reasoning_depth": "minimal",
        },
    ],
    "medium": [
        {
            "id": "medium_1",
            "prompt": "Lee wants to propose marriage to Sierra. He wants to follow the adage that you should spend two months' salary on the ring. He earns $60,000 per year in salary and can save $1000 per month. How long will it take before he can propose to Sierra?",
            "expected_answer": "10",
            "difficulty": "medium",
            "category": "word_problem",
            "max_expected_budget": 160,
            "expected_reasoning_depth": "moderate",
        },
        {
            "id": "medium_2",
            "prompt": "A train travels 120 miles in 2 hours. At this rate, how long will it take to travel 300 miles?",
            "expected_answer": "5",
            "difficulty": "medium",
            "category": "word_problem",
            "max_expected_budget": 160,
            "expected_reasoning_depth": "moderate",
        },
        {
            "id": "medium_3",
            "prompt": "If a rectangle has a length of 12 cm and a width of 8 cm, what is its area and perimeter?",
            "expected_answer": "Area: 96 cm², Perimeter: 40 cm",
            "difficulty": "medium",
            "category": "geometry",
            "max_expected_budget": 160,
            "expected_reasoning_depth": "moderate",
        },
    ],
    "hard": [
        {
            "id": "hard_1",
            "prompt": "Initially Alex, Betty, and Charlie had a total of $444$ peanuts. Charlie had the most peanuts, and Alex had the least. The three numbers of peanuts that each person had formed a geometric progression. Alex eats $5$ of his peanuts, Betty eats $9$ of her peanuts, and Charlie eats $25$ of his peanuts. Now the three numbers of peanuts each person has forms an arithmetic progression. Find the number of peanuts Alex had initially.",
            "expected_answer": "36",
            "difficulty": "hard",
            "category": "algebra",
            "max_expected_budget": 384,
            "expected_reasoning_depth": "extensive",
        },
        {
            "id": "hard_2",
            "prompt": "Find all real solutions to the equation: $x^4 - 10x^2 + 9 = 0$",
            "expected_answer": "x = ±1, ±3",
            "difficulty": "hard",
            "category": "algebra",
            "max_expected_budget": 384,
            "expected_reasoning_depth": "extensive",
        },
        {
            "id": "hard_3",
            "prompt": "A cylindrical tank with radius 3 meters is being filled with water at a rate of 2 cubic meters per minute. If the tank is initially empty, how long will it take for the water level to reach 4 meters high?",
            "expected_answer": "18π minutes (approximately 56.55 minutes)",
            "difficulty": "hard",
            "category": "geometry",
            "max_expected_budget": 384,
            "expected_reasoning_depth": "extensive",
        },
    ],
}

# Legacy queries for backward compatibility
DEFAULT_QUERIES: Dict[str, str] = {
    "easy_1": "What is 1+1?",
    "medium_1": "Lee wants to propose marriage to Sierra. He wants to follow the adage that you should spend two months' salary on the ring. He earns $60,000 per year in salary and can save $1000 per month. How long will it take before he can propose to Sierra?",
    "hard_1": "Initially Alex, Betty, and Charlie had a total of $444$ peanuts. Charlie had the most peanuts, and Alex had the least. The three numbers of peanuts that each person had formed a geometric progression. Alex eats $5$ of his peanuts, Betty eats $9$ of her peanuts, and Charlie eats $25$ of his peanuts. Now the three numbers of peanuts each person has forms an arithmetic progression. Find the number of peanuts Alex had initially.",
}


# ========================= MATRYOSHKA MODEL CLASS ========================= #


class MatryoshkaModel(weave.Model):
    """
    Weave Model wrapper for MatryoshkaEngine that provides structured prediction
    with proper logging and versioning capabilities.
    """

    model_name_or_path: str
    gating_head_path: Optional[str] = None
    budgets: str = ""
    temperature: float = 0.7
    top_p: float = 1.0
    max_new_tokens: int = 512
    use_bf16: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine = None

    @property
    def engine(self) -> MatryoshkaEngine:
        """Lazy initialization of the MatryoshkaEngine"""
        if self._engine is None:
            self._engine = MatryoshkaEngine(
                model_name_or_path=self.model_name_or_path,
                gating_head_path=self.gating_head_path,
                budgets=self.budgets,
                use_bf16=self.use_bf16,
            )
        return self._engine

    @weave.op()
    def predict(
        self, prompt: str, force_budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the Matryoshka engine.

        Args:
            prompt: Input prompt/question
            force_budget: Optional budget override for reasoning tokens

        Returns:
            Dictionary containing reasoning, answer, and metadata
        """
        result = self.engine.generate(
            user_query=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            force_budget=force_budget,
        )

        # Clean and structure the response
        return {
            "reasoning": _clean_assistant_response(result.get("reasoning", "")),
            "answer": _clean_assistant_response(result.get("answer", "")),
            "closed_naturally": result.get("closed_naturally", False),
            "generated_tokens": result.get("generated_tokens", 0),
            "budget_info": _format_gate_info(result.get("budget_info")),
            "raw_reasoning": result.get("reasoning", ""),
            "raw_answer": result.get("answer", ""),
        }


# ========================= UTILITY FUNCTIONS ========================= #


def _format_gate_info(budget_info: Dict) -> Dict:
    """Format budget/gating information for logging"""
    if budget_info is None:
        return {}
    return {
        "source": budget_info.get("source"),
        "anchor": budget_info.get("anchor"),
        "selected_budget": budget_info.get("selected_budget"),
        "probs": budget_info.get("probs"),
    }


def _extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} format commonly used in math problems"""
    import re

    boxed_pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(boxed_pattern, text)
    return matches[-1] if matches else None


def _normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (remove whitespace, convert to lowercase)"""
    if not answer:
        return ""
    return re.sub(r"\s+", " ", answer.strip().lower())


def _extract_numeric_answer(text: str) -> Optional[float]:
    """Extract numeric answer from text for mathematical comparisons"""
    import re

    # Look for numbers (including decimals and fractions)
    number_patterns = [
        r"-?\d+\.?\d*",  # Regular decimals
        r"-?\d+/\d+",  # Fractions
    ]

    for pattern in number_patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                # Try to convert the last match to float
                last_match = matches[-1]
                if "/" in last_match:
                    # Handle fractions
                    num, den = last_match.split("/")
                    return float(num) / float(den)
                else:
                    return float(last_match)
            except (ValueError, ZeroDivisionError):
                continue
    return None


def _clean_assistant_response(text: str) -> str:
    """
    Clean the assistant response by removing chat template artifacts and keeping only the content.

    This handles common chat template patterns like:
    - Special tokens (e.g., <|im_end|>, <|eot_id|>, etc.)
    - Assistant prefixes/suffixes
    - Extra whitespace and newlines
    """
    if not text:
        return ""

    # Remove common special tokens that might appear at the end
    special_tokens = [
        "<|im_end|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "</s>",
        "<eos>",
        "<|endoftext|>",
        "<|assistant|>",
        "<|user|>",
        "<|system|>",
    ]

    cleaned = text
    for token in special_tokens:
        cleaned = cleaned.replace(token, "")

    # Remove excessive whitespace and newlines
    cleaned = cleaned.strip()

    # Remove leading/trailing newlines but preserve internal structure
    while cleaned.startswith("\n"):
        cleaned = cleaned[1:]
    while cleaned.endswith("\n"):
        cleaned = cleaned[:-1]

    return cleaned


# ========================= CUSTOM SCORING FUNCTIONS ========================= #


@weave.op()
def reasoning_quality_scorer(
    expected_reasoning_depth: str, difficulty: str, output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Score the quality and appropriateness of reasoning based on problem difficulty.

    Args:
        expected_reasoning_depth: Expected reasoning depth from dataset
        difficulty: Problem difficulty from dataset
        output: Model output containing reasoning, answer, budget info

    Returns:
        Dictionary with reasoning quality metrics
    """
    reasoning = output.get("reasoning", "")
    reasoning_tokens = output.get("generated_tokens", 0)
    expected_depth = expected_reasoning_depth

    # Basic reasoning presence check
    has_reasoning = len(reasoning.strip()) > 10

    # Depth appropriateness based on expected depth
    depth_scores = {
        "minimal": {"min_tokens": 5, "max_tokens": 50},
        "moderate": {"min_tokens": 20, "max_tokens": 200},
        "extensive": {"min_tokens": 50, "max_tokens": 500},
    }

    expected_range = depth_scores.get(expected_depth, depth_scores["moderate"])
    depth_appropriate = (
        expected_range["min_tokens"] <= reasoning_tokens <= expected_range["max_tokens"]
    )

    # Calculate reasoning density (non-whitespace chars per token)
    reasoning_density = len(reasoning.replace(" ", "")) / max(reasoning_tokens, 1)

    return {
        "has_reasoning": has_reasoning,
        "reasoning_tokens": reasoning_tokens,
        "depth_appropriate": depth_appropriate,
        "reasoning_density": reasoning_density,
        "expected_depth": expected_depth,
        "reasoning_quality_score": (
            0.4 * has_reasoning
            + 0.3 * depth_appropriate
            + 0.3 * min(reasoning_density / 3.0, 1.0)  # Normalize density
        ),
    }


@weave.op()
def token_efficiency_scorer(
    max_expected_budget: int, output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate token efficiency relative to budget and problem complexity.

    Args:
        max_expected_budget: Expected budget from dataset
        output: Model output with token usage and budget info

    Returns:
        Dictionary with efficiency metrics
    """
    generated_tokens = output.get("generated_tokens", 0)
    budget_info = output.get("budget_info", {})
    selected_budget = budget_info.get("selected_budget", -1)

    # Calculate efficiency metrics
    budget_utilization = (
        generated_tokens / selected_budget if selected_budget > 0 else 0
    )

    efficiency_vs_expected = (
        generated_tokens / max_expected_budget if max_expected_budget > 0 else 0
    )

    # Efficiency score (lower is better for token usage)
    efficiency_score = max(0, 1.0 - efficiency_vs_expected)

    return {
        "generated_tokens": generated_tokens,
        "selected_budget": selected_budget,
        "max_expected_budget": max_expected_budget,
        "budget_utilization": budget_utilization,
        "efficiency_vs_expected": efficiency_vs_expected,
        "efficiency_score": efficiency_score,
        "is_efficient": efficiency_vs_expected <= 1.2,  # Within 20% of expected
    }


@weave.op()
def budget_compliance_scorer(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check compliance with budget constraints and closure behavior.

    Args:
        output: Model output with budget and closure information

    Returns:
        Dictionary with budget compliance metrics
    """
    budget_info = output.get("budget_info", {})
    selected_budget = budget_info.get("selected_budget", -1)
    generated_tokens = output.get("generated_tokens", 0)
    closed_naturally = output.get("closed_naturally", False)

    # Budget compliance check
    budget_compliant = (
        selected_budget == -1 or generated_tokens <= selected_budget  # Unlimited budget
    )

    # Closure appropriateness
    closure_appropriate = (
        (selected_budget == 0 and generated_tokens <= 10)  # Depth 0 should be minimal
        or (
            selected_budget > 0 and closed_naturally
        )  # Should close naturally within budget
        or selected_budget == -1  # Unlimited budget
    )

    return {
        "selected_budget": selected_budget,
        "generated_tokens": generated_tokens,
        "closed_naturally": closed_naturally,
        "budget_compliant": budget_compliant,
        "closure_appropriate": closure_appropriate,
        "budget_compliance_score": (0.6 * budget_compliant + 0.4 * closure_appropriate),
    }


@weave.op()
def answer_accuracy_scorer(
    expected_answer: str, output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate answer accuracy using multiple comparison methods.

    Args:
        expected_answer: Expected answer from dataset
        output: Model output with answer

    Returns:
        Dictionary with accuracy metrics
    """
    model_answer = output.get("answer", "")

    # Extract boxed answer if present
    boxed_answer = _extract_boxed_answer(model_answer)
    answer_to_check = boxed_answer if boxed_answer else model_answer

    # Normalize for comparison
    normalized_expected = _normalize_answer(expected_answer)
    normalized_answer = _normalize_answer(answer_to_check)

    # Exact match
    exact_match = normalized_expected == normalized_answer

    # Substring match (for partial credit)
    substring_match = (
        normalized_expected in normalized_answer
        or normalized_answer in normalized_expected
    )

    # Numeric comparison for math problems
    numeric_match = False
    expected_num = _extract_numeric_answer(expected_answer)
    answer_num = _extract_numeric_answer(answer_to_check)

    if expected_num is not None and answer_num is not None:
        numeric_match = abs(expected_num - answer_num) < 1e-6

    # Overall accuracy score
    accuracy_score = (
        1.0
        if exact_match
        else 0.8 if numeric_match else 0.3 if substring_match else 0.0
    )

    return {
        "expected_answer": expected_answer,
        "model_answer": answer_to_check,
        "exact_match": exact_match,
        "substring_match": substring_match,
        "numeric_match": numeric_match,
        "accuracy_score": accuracy_score,
        "has_boxed_answer": boxed_answer is not None,
    }


# ========================= DATASET CREATION ========================= #


def create_weave_datasets() -> Dict[str, weave.Dataset]:
    """Create Weave datasets from structured data organized by difficulty."""
    datasets = {}

    for difficulty, examples in STRUCTURED_DATASETS.items():
        # Convert examples to the format expected by weave.Dataset
        dataset_rows = []
        for example in examples:
            dataset_rows.append(
                {
                    "id": example["id"],
                    "prompt": example["prompt"],
                    "expected_answer": example["expected_answer"],
                    "difficulty": example["difficulty"],
                    "category": example["category"],
                    "max_expected_budget": example["max_expected_budget"],
                    "expected_reasoning_depth": example["expected_reasoning_depth"],
                }
            )

        datasets[difficulty] = weave.Dataset(
            name=f"matryoshka_{difficulty}_eval", rows=dataset_rows
        )

    # Create combined dataset
    all_rows = []
    for examples in STRUCTURED_DATASETS.values():
        for example in examples:
            all_rows.append(
                {
                    "id": example["id"],
                    "prompt": example["prompt"],
                    "expected_answer": example["expected_answer"],
                    "difficulty": example["difficulty"],
                    "category": example["category"],
                    "max_expected_budget": example["max_expected_budget"],
                    "expected_reasoning_depth": example["expected_reasoning_depth"],
                }
            )

    datasets["combined"] = weave.Dataset(name="matryoshka_combined_eval", rows=all_rows)

    return datasets


# ========================= MULTI-DEPTH EVALUATION ========================= #


@weave.op()
def multi_depth_evaluation(
    model: MatryoshkaModel, dataset: weave.Dataset, budgets: List[int] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance across different reasoning budgets.

    Args:
        model: MatryoshkaModel instance
        dataset: Dataset to evaluate on
        budgets: List of budgets to test (default: DEFAULT_BUDGETS)

    Returns:
        Dictionary with results for each budget
    """
    if budgets is None:
        budgets = DEFAULT_BUDGETS

    results = {}

    for budget in budgets:
        print(f"🔍 Evaluating with budget: {budget}")
        budget_results = []

        for example in dataset.rows:
            # Generate prediction with specific budget
            prediction = model.predict(
                prompt=example["prompt"], force_budget=budget if budget >= 0 else None
            )

            # Score the prediction
            reasoning_score = reasoning_quality_scorer(
                expected_reasoning_depth=example["expected_reasoning_depth"],
                difficulty=example["difficulty"],
                output=prediction,
            )
            efficiency_score = token_efficiency_scorer(
                max_expected_budget=example["max_expected_budget"], output=prediction
            )
            compliance_score = budget_compliance_scorer(output=prediction)
            accuracy_score = answer_accuracy_scorer(
                expected_answer=example["expected_answer"], output=prediction
            )

            budget_results.append(
                {
                    "example_id": example["id"],
                    "budget": budget,
                    "prediction": prediction,
                    "scores": {
                        "reasoning": reasoning_score,
                        "efficiency": efficiency_score,
                        "compliance": compliance_score,
                        "accuracy": accuracy_score,
                    },
                }
            )

        # Calculate aggregate metrics for this budget
        accuracy_scores = [
            r["scores"]["accuracy"]["accuracy_score"] for r in budget_results
        ]
        efficiency_scores = [
            r["scores"]["efficiency"]["efficiency_score"] for r in budget_results
        ]
        compliance_scores = [
            r["scores"]["compliance"]["budget_compliance_score"] for r in budget_results
        ]
        reasoning_scores = [
            r["scores"]["reasoning"]["reasoning_quality_score"] for r in budget_results
        ]

        results[f"budget_{budget}"] = {
            "budget": budget,
            "individual_results": budget_results,
            "aggregate_metrics": {
                "mean_accuracy": statistics.mean(accuracy_scores),
                "mean_efficiency": statistics.mean(efficiency_scores),
                "mean_compliance": statistics.mean(compliance_scores),
                "mean_reasoning_quality": statistics.mean(reasoning_scores),
                "total_examples": len(budget_results),
                "exact_matches": sum(
                    1 for r in budget_results if r["scores"]["accuracy"]["exact_match"]
                ),
                "budget_compliant": sum(
                    1
                    for r in budget_results
                    if r["scores"]["compliance"]["budget_compliant"]
                ),
                "natural_closures": sum(
                    1 for r in budget_results if r["prediction"]["closed_naturally"]
                ),
            },
        }

    return results


# ========================= COMPREHENSIVE BENCHMARKING ========================= #


@weave.op()
async def comprehensive_benchmark(
    model: MatryoshkaModel,
    datasets: Dict[str, weave.Dataset] = None,
    budgets: List[int] = None,
    include_multi_depth: bool = True,
) -> Dict[str, Any]:
    """
    Run comprehensive benchmarking across multiple datasets and budgets.

    Args:
        model: MatryoshkaModel instance
        datasets: Dictionary of datasets to evaluate on
        budgets: List of budgets for multi-depth evaluation
        include_multi_depth: Whether to include multi-depth analysis

    Returns:
        Comprehensive benchmark results with statistical analysis
    """
    if datasets is None:
        datasets = create_weave_datasets()

    if budgets is None:
        budgets = DEFAULT_BUDGETS

    benchmark_results = {
        "model_info": {
            "model_path": model.model_name_or_path,
            "gating_head": model.gating_head_path,
            "budgets": model.budgets,
            "temperature": model.temperature,
            "max_new_tokens": model.max_new_tokens,
        },
        "dataset_results": {},
        "multi_depth_results": {},
        "summary_statistics": {},
    }

    # Standard evaluation on each dataset
    all_scorers = [
        reasoning_quality_scorer,
        token_efficiency_scorer,
        budget_compliance_scorer,
        answer_accuracy_scorer,
    ]

    for dataset_name, dataset in datasets.items():
        if dataset_name == "combined":
            continue  # Skip combined for individual dataset evaluation

        print(f"📊 Evaluating on {dataset_name} dataset...")

        # Create evaluation
        evaluation = weave.Evaluation(
            dataset=dataset,
            scorers=all_scorers,
            name=f"matryoshka_{dataset_name}_benchmark",
        )

        # Run evaluation
        eval_results = await evaluation.evaluate(model)
        benchmark_results["dataset_results"][dataset_name] = eval_results

    # Multi-depth evaluation on combined dataset
    if include_multi_depth and "combined" in datasets:
        print("🔍 Running multi-depth evaluation...")
        multi_depth_results = multi_depth_evaluation(
            model=model, dataset=datasets["combined"], budgets=budgets
        )
        benchmark_results["multi_depth_results"] = multi_depth_results

    # Calculate summary statistics
    benchmark_results["summary_statistics"] = _calculate_summary_statistics(
        benchmark_results
    )

    return benchmark_results


def _calculate_summary_statistics(benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics across all evaluations."""
    summary = {
        "overall_accuracy": 0.0,
        "overall_efficiency": 0.0,
        "overall_compliance": 0.0,
        "overall_reasoning_quality": 0.0,
        "budget_performance": {},
        "difficulty_performance": {},
        "category_performance": {},
    }

    # Aggregate across dataset results
    dataset_results = benchmark_results.get("dataset_results", {})
    if dataset_results:
        all_accuracy = []
        all_efficiency = []
        all_compliance = []
        all_reasoning = []

        for dataset_name, results in dataset_results.items():
            # Extract scores from evaluation results
            # Note: This assumes weave.Evaluation returns results in a specific format
            # You may need to adjust based on actual Weave evaluation result structure
            if hasattr(results, "scores") or "scores" in results:
                scores = results.get("scores", results)
                if isinstance(scores, dict):
                    for score_name, score_value in scores.items():
                        if "accuracy" in score_name.lower():
                            all_accuracy.append(score_value)
                        elif "efficiency" in score_name.lower():
                            all_efficiency.append(score_value)
                        elif "compliance" in score_name.lower():
                            all_compliance.append(score_value)
                        elif "reasoning" in score_name.lower():
                            all_reasoning.append(score_value)

        if all_accuracy:
            summary["overall_accuracy"] = statistics.mean(all_accuracy)
        if all_efficiency:
            summary["overall_efficiency"] = statistics.mean(all_efficiency)
        if all_compliance:
            summary["overall_compliance"] = statistics.mean(all_compliance)
        if all_reasoning:
            summary["overall_reasoning_quality"] = statistics.mean(all_reasoning)

    # Aggregate multi-depth results
    multi_depth_results = benchmark_results.get("multi_depth_results", {})
    if multi_depth_results:
        for budget_key, budget_data in multi_depth_results.items():
            budget = budget_data.get("budget", "unknown")
            metrics = budget_data.get("aggregate_metrics", {})

            summary["budget_performance"][budget] = {
                "accuracy": metrics.get("mean_accuracy", 0.0),
                "efficiency": metrics.get("mean_efficiency", 0.0),
                "compliance": metrics.get("mean_compliance", 0.0),
                "reasoning_quality": metrics.get("mean_reasoning_quality", 0.0),
                "exact_match_rate": metrics.get("exact_matches", 0)
                / max(metrics.get("total_examples", 1), 1),
                "compliance_rate": metrics.get("budget_compliant", 0)
                / max(metrics.get("total_examples", 1), 1),
                "natural_closure_rate": metrics.get("natural_closures", 0)
                / max(metrics.get("total_examples", 1), 1),
            }

    return summary


# ========================= ENHANCED MAIN FUNCTIONS ========================= #


async def run_enhanced_evaluation(
    model: MatryoshkaModel,
    enhanced_eval: bool = True,
    multi_depth_eval: bool = False,
    budgets: List[int] = None,
) -> Dict[str, Any]:
    """
    Run enhanced evaluation using Weave's evaluation framework.

    Args:
        model: MatryoshkaModel instance
        enhanced_eval: Whether to run enhanced structured evaluation
        multi_depth_eval: Whether to run multi-depth evaluation
        budgets: List of budgets for multi-depth evaluation

    Returns:
        Dictionary with evaluation results
    """
    results = {}

    if enhanced_eval:
        print("🚀 Running enhanced structured evaluation...")
        datasets = create_weave_datasets()

        # Run comprehensive benchmark
        benchmark_results = await comprehensive_benchmark(
            model=model,
            datasets=datasets,
            budgets=budgets,
            include_multi_depth=multi_depth_eval,
        )
        results["enhanced_evaluation"] = benchmark_results

        # Print summary
        summary = benchmark_results["summary_statistics"]
        print(f"\n{'='*60}")
        print(f"📊 ENHANCED EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.3f}")
        print(f"Overall Efficiency: {summary['overall_efficiency']:.3f}")
        print(f"Overall Compliance: {summary['overall_compliance']:.3f}")
        print(f"Overall Reasoning Quality: {summary['overall_reasoning_quality']:.3f}")

        # Print detailed budget selections for each dataset
        print(f"\n🎯 DETAILED BUDGET SELECTIONS:")
        dataset_results = benchmark_results.get("dataset_results", {})
        for dataset_name, eval_result in dataset_results.items():
            print(f"\n📊 {dataset_name.upper()} Dataset:")
            # Note: The exact structure of eval_result depends on Weave's return format
            # This is a placeholder - you may need to adjust based on actual structure
            if hasattr(eval_result, "rows") or isinstance(eval_result, dict):
                print(
                    f"  Evaluation completed - check Weave dashboard for detailed results"
                )

        if multi_depth_eval and summary["budget_performance"]:
            print(f"\n📈 BUDGET PERFORMANCE:")
            for budget, perf in summary["budget_performance"].items():
                print(
                    f"  Budget {budget}: Acc={perf['accuracy']:.3f}, Eff={perf['efficiency']:.3f}"
                )

    return results


def create_weave_op(query_name: str):
    """Create a Weave op function with custom name conditionally based on availability."""
    if WEAVE_AVAILABLE:

        @weave.op(name=query_name)
        def query_evaluation(
            engine_name: str,
            prompt: str,
            temperature: float,
            top_p: float,
            max_new_tokens: int,
            force_budget: Optional[int],
            reasoning: str,
            answer: str,
            closed_naturally: bool,
            generated_tokens: int,
            budget_info: Dict,
        ) -> Dict:
            return {
                "engine_name": engine_name,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "force_budget": force_budget,
                "reasoning": reasoning,
                "answer": answer,
                "closed_naturally": closed_naturally,
                "generated_tokens": generated_tokens,
                "budget_info": budget_info,
            }

        return query_evaluation
    else:
        return None


def run_one(
    engine: MatryoshkaEngine,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    force_budget: Optional[int],
) -> Dict:
    res = engine.generate(
        user_query=prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        force_budget=force_budget,
    )
    # Clean the assistant response to remove chat template artifacts
    raw_reasoning = res.get("reasoning", "")
    raw_answer = res.get("answer", "")

    out = {
        "prompt": prompt,
        "reasoning": _clean_assistant_response(raw_reasoning),
        "answer": _clean_assistant_response(raw_answer),
        "closed_naturally": res.get("closed_naturally", False),
        "generated_tokens": res.get("generated_tokens", 0),
        "budget_info": _format_gate_info(res.get("budget_info")),
    }
    return out


def save_results_locally(results: List[Dict], model: str, project: str) -> str:
    """Save results to a local JSON file as fallback."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.replace("/", "_").replace("\\", "_")
    filename = f"qualitative_eval_{model_name}_{timestamp}.json"

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    # Add metadata to the results
    output_data = {
        "metadata": {
            "model": model,
            "project": project,
            "timestamp": timestamp,
            "total_queries": len(results),
        },
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return filepath


def main(
    model: str,
    gating_head: Optional[str] = None,
    project: str = "matryoshka-reasoning",
    budgets: str = "",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_new_tokens: int = 512,
    force_budget: int = -1,
    use_bf16: bool = True,
    # Enhanced evaluation parameters
    enhanced_eval: bool = True,
    multi_depth_eval: bool = False,
    budget_list: str = "",  # Comma-separated list for multi-depth eval
):
    global WEAVE_AVAILABLE

    # Try to initialize Weave
    try:
        weave.init(project)  # 🐝
        WEAVE_AVAILABLE = True
        print(f"✅ Weave initialized successfully for project: {project}")
    except Exception as e:
        print(f"⚠️  Weave initialization failed: {e}")
        print("Continuing without Weave logging...")
        WEAVE_AVAILABLE = False

    # Parse budget list for multi-depth evaluation
    eval_budgets = None
    if budget_list:
        try:
            eval_budgets = [int(b.strip()) for b in budget_list.split(",")]
        except ValueError:
            print(f"⚠️  Invalid budget list format: {budget_list}. Using defaults.")
            eval_budgets = DEFAULT_BUDGETS

    # Create MatryoshkaModel for enhanced evaluation
    matryoshka_model = MatryoshkaModel(
        model_name_or_path=model,
        gating_head_path=gating_head,
        budgets=budgets,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        use_bf16=use_bf16,
    )

    # Run enhanced evaluation if requested
    enhanced_results = {}
    if enhanced_eval or multi_depth_eval:
        if WEAVE_AVAILABLE:
            enhanced_results = asyncio.run(
                run_enhanced_evaluation(
                    model=matryoshka_model,
                    enhanced_eval=enhanced_eval,
                    multi_depth_eval=multi_depth_eval,
                    budgets=eval_budgets,
                )
            )
        else:
            print(
                "⚠️  Enhanced evaluation requires Weave. Falling back to legacy evaluation."
            )

    # Legacy evaluation (for backward compatibility)
    engine = MatryoshkaEngine(
        model_name_or_path=model,
        gating_head_path=gating_head,
        budgets=budgets,
        use_bf16=use_bf16,
    )

    results = []
    weave_logged_count = 0

    # Determine which queries to run
    if enhanced_eval and not multi_depth_eval:
        print(
            f"🚀 Enhanced evaluation completed. Running legacy evaluation for comparison..."
        )
        print(
            f"🚀 Running qualitative evaluation on {len(DEFAULT_QUERIES)} legacy queries..."
        )
    else:
        print(f"🚀 Running qualitative evaluation on {len(DEFAULT_QUERIES)} queries...")

    for i, (query_name, query_text) in enumerate(DEFAULT_QUERIES.items(), 1):
        print(
            f"\n📝 Query {i}/{len(DEFAULT_QUERIES)} ({query_name}): {query_text[:50]}{'...' if len(query_text) > 50 else ''}"
        )

        out = run_one(
            engine=engine,
            prompt=query_text,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            force_budget=(
                force_budget if force_budget is not None and force_budget >= 0 else None
            ),
        )
        results.append(out)

        # Log to Weave if available
        if WEAVE_AVAILABLE:
            weave_op_func = create_weave_op(query_name)
            if weave_op_func:
                try:
                    weave_op_func(
                        engine_name=model,
                        prompt=out["prompt"],
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        force_budget=force_budget,
                        reasoning=out["reasoning"],
                        answer=out["answer"],
                        closed_naturally=out["closed_naturally"],
                        generated_tokens=out["generated_tokens"],
                        budget_info=out["budget_info"],
                    )
                    weave_logged_count += 1
                    print(f"  ✅ Logged to Weave as '{query_name}'")
                except Exception as e:
                    print(f"  ⚠️  Failed to log to Weave: {e}")

        print(
            f"  🎯 Answer: {out['answer'][:100]}{'...' if len(out['answer']) > 100 else ''}"
        )
        print(f"  🧠 Reasoning tokens: {out['generated_tokens']}")
        if out["budget_info"].get("selected_budget"):
            print(f"  📊 Selected budget: {out['budget_info']['selected_budget']}")

    # Combine results for saving
    combined_results = {
        "legacy_evaluation": results,
        "enhanced_evaluation": enhanced_results if enhanced_results else None,
        "evaluation_config": {
            "enhanced_eval": enhanced_eval,
            "multi_depth_eval": multi_depth_eval,
            "eval_budgets": eval_budgets,
            "model_config": {
                "model_path": model,
                "gating_head": gating_head,
                "budgets": budgets,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "force_budget": force_budget,
            },
        },
    }

    # Save results locally as fallback or primary storage
    local_file = save_results_locally(results, model, project)

    # Save enhanced results if available
    if enhanced_results:
        enhanced_file = save_enhanced_results(combined_results, model, project)
        print(f"📄 Enhanced results saved to: {enhanced_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"📝 Legacy queries processed: {len(results)}")
    print(f"📄 Legacy results saved to: {local_file}")

    if WEAVE_AVAILABLE:
        print(f"🐝 Weave logs: {weave_logged_count}/{len(results)} successful")
        print(f"🌐 Weave project: {project}")

        if enhanced_eval or multi_depth_eval:
            print(f"🚀 Enhanced evaluation completed with Weave integration")
    else:
        print(f"⚠️  Weave unavailable - results saved locally only")

    # Print a compact JSON summary for legacy results
    if (
        not enhanced_eval
    ):  # Only print detailed legacy results if not running enhanced eval
        print(f"\n{'='*60}")
        print(f"📋 DETAILED LEGACY RESULTS")
        print(f"{'='*60}")
        print(json.dumps(results, indent=2, ensure_ascii=False))


def save_enhanced_results(
    combined_results: Dict[str, Any], model: str, project: str
) -> str:
    """Save enhanced evaluation results to a local JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.replace("/", "_").replace("\\", "_")
    filename = f"enhanced_eval_{model_name}_{timestamp}.json"

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    # Add metadata to the results
    output_data = {
        "metadata": {
            "model": model,
            "project": project,
            "timestamp": timestamp,
            "evaluation_type": "enhanced_matryoshka_evaluation",
        },
        "results": combined_results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return filepath


if __name__ == "__main__":
    fire.Fire(main)
