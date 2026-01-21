"""Demo 4: Error tracing and debugging workflows.

This demo shows how LangSmith captures errors in traces for
debugging purposes, including stack traces and error context.

Key concepts demonstrated:
- Automatic error capture in traces
- Stack trace logging
- Error context preservation
- Debugging workflow with LangSmith
- Handling different error types
"""

from __future__ import annotations

from typing import Any

from langsmith import traceable

from shared.logger import logger
from shared.observability import add_run_metadata, add_run_tags, is_tracing_enabled


# region Private Functions


@traceable(name="Error-Prone Operation", run_type="chain")
async def error_prone_operation(
    operation_type: str,
    *,
    should_fail: bool = True,
) -> dict[str, Any]:
    """An operation that may fail for demonstration.

    Args:
        operation_type: Type of operation to attempt.
        should_fail: Whether the operation should fail.

    Returns:
        Result dict if successful.

    Raises:
        Various exceptions depending on operation_type.
    """
    add_run_metadata(
        {
            "operation_type": operation_type,
            "expected_failure": should_fail,
        }
    )
    add_run_tags([f"operation:{operation_type}"])

    if not should_fail:
        return {"status": "success", "operation": operation_type}

    if operation_type == "validation":
        add_run_metadata({"error_stage": "input_validation"})
        msg = "Input validation failed: required field 'data' is missing"
        raise ValueError(msg)

    if operation_type == "timeout":
        add_run_metadata({"error_stage": "network_request"})
        msg = "Request timed out after 30 seconds waiting for external service"
        raise TimeoutError(msg)

    if operation_type == "permission":
        add_run_metadata({"error_stage": "authorization"})
        msg = "Permission denied: user lacks required 'admin' role"
        raise PermissionError(msg)

    if operation_type == "runtime":
        add_run_metadata({"error_stage": "processing"})
        msg = "Unexpected runtime error during data processing"
        raise RuntimeError(msg)

    # Division by zero for unexpected errors
    add_run_metadata({"error_stage": "calculation"})
    return {"result": 1 / 0}


@traceable(name="Nested Error Chain", run_type="chain")
async def nested_error_chain() -> dict[str, Any]:
    """Demonstrate error propagation through nested calls.

    This shows how errors in nested traced functions
    appear in the trace hierarchy.

    Returns:
        Never returns - always raises.

    Raises:
        RuntimeError: Propagated from inner function.
    """
    add_run_metadata({"depth": 0, "stage": "outer"})

    try:
        return await inner_operation()
    except Exception as e:
        add_run_metadata({"caught_error": str(e)})
        msg = f"Outer operation failed due to inner error: {e}"
        raise RuntimeError(msg) from e


@traceable(name="Inner Operation", run_type="tool")
async def inner_operation() -> dict[str, Any]:
    """Inner operation that fails.

    Returns:
        Never returns.

    Raises:
        ValueError: Always raised.
    """
    add_run_metadata({"depth": 1, "stage": "inner"})

    # Simulate some work before failing
    add_run_metadata({"progress": "50%"})

    msg = "Inner operation encountered invalid state"
    raise ValueError(msg)


@traceable(name="Recoverable Operation", run_type="chain")
async def recoverable_operation() -> dict[str, Any]:
    """Demonstrate graceful error handling with tracing.

    Shows how to trace both the error and recovery.

    Returns:
        Result with recovery information.
    """
    add_run_metadata({"attempt": 1})

    try:
        return await flaky_operation(fail=True)
    except ValueError:
        add_run_metadata({"recovery_triggered": True})
        add_run_tags(["recovered"])

        # Retry with different parameters
        add_run_metadata({"attempt": 2})
        result = await flaky_operation(fail=False)
        result["recovered"] = True
        return result


@traceable(name="Flaky Operation", run_type="tool")
async def flaky_operation(*, fail: bool = True) -> dict[str, Any]:
    """Operation that sometimes fails.

    Args:
        fail: Whether to fail.

    Returns:
        Success result if not failing.

    Raises:
        ValueError: If fail is True.
    """
    add_run_metadata({"will_fail": fail})

    if fail:
        msg = "Flaky operation failed (simulated)"
        raise ValueError(msg)

    return {"status": "success", "data": "operation_result"}


# endregion


# region Public Functions


async def run_error_tracing_demo() -> None:
    """Run the error tracing demo.

    Demonstrates:
    - How errors are captured in traces
    - Stack trace visibility
    - Error context and metadata
    - Nested error propagation
    - Recovery patterns with tracing
    """
    print(f"\nTracing enabled: {is_tracing_enabled()}")

    if is_tracing_enabled():
        print("Errors will be captured in LangSmith traces")
    else:
        print("Tracing disabled - errors will occur but won't be traced")

    # Demo 1: Different error types
    print("\n--- Demo 1: Different Error Types ---")

    error_types = ["validation", "timeout", "permission", "runtime"]

    for error_type in error_types:
        print(f"\nTriggering {error_type} error...")
        try:
            await error_prone_operation(error_type, should_fail=True)
        except Exception as e:
            print(f"  Caught: {type(e).__name__}: {e}")
            logger.debug(f"Error traced: {error_type}", extra={"error": str(e)})

    # Demo 2: Nested error chain
    print("\n--- Demo 2: Nested Error Chain ---")
    print("\nTriggering nested error chain...")

    try:
        await nested_error_chain()
    except RuntimeError as e:
        print(f"  Caught outer error: {e}")
        if e.__cause__:
            print(f"  Caused by: {type(e.__cause__).__name__}: {e.__cause__}")

    # Demo 3: Recovery with tracing
    print("\n--- Demo 3: Recovery with Tracing ---")
    print("\nAttempting recoverable operation...")

    try:
        result = await recoverable_operation()
        print(f"  Operation succeeded after recovery: {result}")
    except Exception as e:
        print(f"  Recovery failed: {e}")

    # Demo 4: Successful operation (for comparison)
    print("\n--- Demo 4: Successful Operation (for comparison) ---")
    print("\nRunning successful operation...")

    try:
        result = await error_prone_operation("success", should_fail=False)
        print(f"  Success: {result}")
    except Exception as e:
        print(f"  Unexpected error: {e}")

    # Debugging instructions
    print("\n--- Debugging Errors in LangSmith ---")
    print("\nWhen viewing error traces in LangSmith:")
    print("  1. Failed spans are highlighted in red")
    print("  2. Click on a failed span to see the error message")
    print("  3. The 'Error' tab shows the full stack trace")
    print("  4. Metadata shows context at time of error")
    print("  5. Use filters: is_error = true")

    print("\nUseful filter queries:")
    print("  - Find all errors: is_error = true")
    print("  - Find validation errors: has(tags, 'operation:validation')")
    print("  - Find recovered operations: has(tags, 'recovered')")

    if is_tracing_enabled():
        print("\nView error traces at: https://smith.langchain.com/")
        print("Filter by: is_error = true")


# endregion
