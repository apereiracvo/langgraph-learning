"""Demo 2: Custom spans with @traceable decorator.

This demo shows how to explicitly trace custom functions
using the @traceable decorator for fine-grained control.

Key concepts demonstrated:
- @traceable decorator usage
- Nested trace spans
- Custom run types (chain, tool, llm)
- Dynamic metadata addition within spans
"""

from __future__ import annotations

import asyncio
from typing import Any

from langsmith import traceable

from shared.logger import logger
from shared.observability import add_run_metadata, is_tracing_enabled


# region Private Functions


@traceable(name="Data Processing Pipeline", run_type="chain")
async def process_data_pipeline(data: dict[str, Any]) -> dict[str, Any]:
    """Process data through a traced pipeline.

    This function demonstrates a multi-step pipeline where each
    step is traced as a nested span.

    Args:
        data: Input data to process.

    Returns:
        Processed data with validation, transformation, and enrichment.
    """
    logger.info("Starting data pipeline", extra={"input": data})

    # Add pipeline metadata
    add_run_metadata(
        {
            "pipeline_version": "1.0.0",
            "input_keys": list(data.keys()),
        }
    )

    # Step 1: Validate
    validated = await validate_data(data)

    # Step 2: Transform
    transformed = await transform_data(validated)

    # Step 3: Enrich
    enriched = await enrich_data(transformed)

    # Add completion metadata
    add_run_metadata(
        {
            "pipeline_completed": True,
            "output_keys": list(enriched.keys()),
        }
    )

    return enriched


@traceable(name="Validate Input", run_type="tool")
async def validate_data(data: dict[str, Any]) -> dict[str, Any]:
    """Validate input data.

    Args:
        data: Data to validate.

    Returns:
        Validated data with validation status.
    """
    # Simulate async validation
    await asyncio.sleep(0.05)

    is_valid = bool(data.get("value"))
    add_run_metadata(
        {
            "validation_passed": is_valid,
            "checked_fields": ["value"],
        }
    )

    return {
        **data,
        "validated": True,
        "validation_status": "passed" if is_valid else "failed",
    }


@traceable(name="Transform Data", run_type="tool")
async def transform_data(data: dict[str, Any]) -> dict[str, Any]:
    """Transform data format.

    Args:
        data: Data to transform.

    Returns:
        Transformed data with doubled value.
    """
    # Simulate async transformation
    await asyncio.sleep(0.05)

    value = data.get("value", 0)
    transformed_value = value * 2  # Simple transformation

    add_run_metadata(
        {
            "original_value": value,
            "transformed_value": transformed_value,
            "transformation_type": "double",
        }
    )

    return {
        **data,
        "value": transformed_value,
        "transformed": True,
    }


@traceable(name="Enrich Data", run_type="tool")
async def enrich_data(data: dict[str, Any]) -> dict[str, Any]:
    """Enrich data with additional context.

    Args:
        data: Data to enrich.

    Returns:
        Enriched data with source and timestamp.
    """
    # Simulate async enrichment (e.g., external API call)
    await asyncio.sleep(0.05)

    add_run_metadata(
        {
            "enrichment_source": "demo",
            "fields_added": ["enrichment"],
        }
    )

    return {
        **data,
        "enriched": True,
        "enrichment": {
            "source": "demo_custom_spans",
            "timestamp": "2024-01-01T00:00:00Z",
            "version": "1.0",
        },
    }


@traceable(name="Parallel Processing", run_type="chain")
async def process_parallel(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process multiple items in parallel with tracing.

    Demonstrates how asyncio.gather works with traced functions,
    creating parallel child spans.

    Args:
        items: List of items to process.

    Returns:
        List of processed items.
    """
    add_run_metadata(
        {
            "item_count": len(items),
            "processing_mode": "parallel",
        }
    )

    # Process all items in parallel
    tasks = [process_single_item(item) for item in items]
    results = await asyncio.gather(*tasks)

    add_run_metadata(
        {
            "completed_count": len(results),
        }
    )

    return list(results)


@traceable(name="Process Single Item", run_type="tool")
async def process_single_item(item: dict[str, Any]) -> dict[str, Any]:
    """Process a single item.

    Args:
        item: Item to process.

    Returns:
        Processed item.
    """
    item_id = item.get("id", "unknown")
    add_run_metadata({"item_id": item_id})

    await asyncio.sleep(0.02)  # Simulate processing

    return {
        **item,
        "processed": True,
    }


# endregion


# region Public Functions


async def run_custom_spans_demo() -> None:
    """Run the custom spans demo.

    Demonstrates:
    - @traceable decorator on functions
    - Nested trace spans
    - Custom run types
    - Dynamic metadata addition
    - Parallel processing with traced functions
    """
    print(f"\nTracing enabled: {is_tracing_enabled()}")

    if is_tracing_enabled():
        print("Custom spans will be visible in LangSmith")
    else:
        print("Tracing disabled - functions will run but no traces sent")

    # Demo 1: Sequential Pipeline
    print("\n--- Sequential Pipeline Demo ---")

    input_data = {"value": 42, "name": "demo_item"}
    print(f"\nInput data: {input_data}")
    print("\n[Processing through traced pipeline...]")

    result = await process_data_pipeline(input_data)

    print(f"\nOutput data: {result}")

    # Demo 2: Parallel Processing
    print("\n--- Parallel Processing Demo ---")

    items = [
        {"id": "item_1", "value": 10},
        {"id": "item_2", "value": 20},
        {"id": "item_3", "value": 30},
    ]
    print(f"\nInput items: {items}")
    print("\n[Processing items in parallel...]")

    parallel_results = await process_parallel(items)

    print(f"\nProcessed items: {parallel_results}")

    # Summary
    print("\n--- Trace Hierarchy Created ---")
    print("\nSequential Pipeline:")
    print("  Data Processing Pipeline (chain)")
    print("    -> Validate Input (tool)")
    print("    -> Transform Data (tool)")
    print("    -> Enrich Data (tool)")

    print("\nParallel Processing:")
    print("  Parallel Processing (chain)")
    print("    -> Process Single Item (tool) x 3 [parallel]")

    if is_tracing_enabled():
        print("\nView the trace hierarchy at: https://smith.langchain.com/")
        print("Look for nested spans showing the pipeline structure")


# endregion
