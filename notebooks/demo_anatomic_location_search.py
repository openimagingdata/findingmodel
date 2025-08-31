"""Manual testing script for anatomic location search with real LanceDB."""

import asyncio

from findingmodel.tools.anatomic_location_search import find_anatomic_locations


async def test_pcl_tear() -> None:
    """Test with PCL tear example from PRD."""
    print("🔍 Testing PCL tear...")
    try:
        result = await find_anatomic_locations(
            finding_name="posterior cruciate ligament tear",
            description="Injury to the posterior cruciate ligament (PCL) in the knee joint, resulting in pain and instability",
        )
        print(f"✅ Primary: {result.primary_location.concept_text}")
        print(f"🔄 Alternates: {[alt.concept_text for alt in result.alternate_locations]}")
        print(f"💭 Reasoning: {result.reasoning}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print()


async def test_heart_chamber_enlargement() -> None:
    """Test with enlarged heart chambers."""
    print("🔍 Testing heart chamber enlargement...")
    try:
        result = await find_anatomic_locations(
            finding_name="cardiac chamber enlargement", description="Abnormally large heart chambers"
        )
        print(f"✅ Primary: {result.primary_location.concept_text}")
        print(f"🔄 Alternates: {[alt.concept_text for alt in result.alternate_locations]}")
        print(f"💭 Reasoning: {result.reasoning}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print()


async def test_pneumothorax() -> None:
    """Test with pneumothorax."""
    print("🔍 Testing pneumothorax...")
    try:
        result = await find_anatomic_locations(
            finding_name="pneumothorax", description="Presence of air in the pleural space causing lung collapse"
        )
        print(f"✅ Primary: {result.primary_location.concept_text}")
        print(f"🔄 Alternates: {[alt.concept_text for alt in result.alternate_locations]}")
        print(f"💭 Reasoning: {result.reasoning}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print()


async def test_ependymoma() -> None:
    """Test with brain tumor."""
    print("🔍 Testing brain tumor...")
    try:
        result = await find_anatomic_locations(
            finding_name="ependymoma",
            description="Abnormal growth of ependymal cells lining the ventricles of the brain or spinal cord",
        )
        print(f"✅ Primary: {result.primary_location.concept_text}")
        print(f"🔄 Alternates: {[alt.concept_text for alt in result.alternate_locations]}")
        print(f"💭 Reasoning: {result.reasoning}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print()


async def test_edge_cases() -> None:
    """Test edge cases."""
    print("🔍 Testing edge cases...")

    # Test with minimal input
    print("  📝 Testing minimal input...")
    try:
        result = await find_anatomic_locations("headache")
        print(f"    ✅ Primary: {result.primary_location.concept_text}")
        print(f"    💭 Reasoning: {result.reasoning[:100]}...")
    except Exception as e:
        print(f"    ❌ Error: {e}")

    # Test with very specific finding
    print("  📝 Testing specific finding...")
    try:
        result = await find_anatomic_locations(
            finding_name="anterior cruciate ligament partial tear",
            description="Partial disruption of the ACL fibers in the knee joint",
        )
        print(f"    ✅ Primary: {result.primary_location.concept_text}")
        print(f"    💭 Reasoning: {result.reasoning[:100]}...")
    except Exception as e:
        print(f"    ❌ Error: {e}")

    print()


async def test_performance() -> None:
    """Test performance with different model configurations."""
    print("🔍 Testing performance with different models...")

    test_cases = [
        ("Fast models", "gpt-4o-mini", "gpt-4o-mini"),
        ("Mixed models", "gpt-4o-mini", "gpt-4o"),
        ("Premium models", "gpt-4o", "gpt-4o"),
    ]

    for desc, search_model, matching_model in test_cases:
        print(f"  📝 {desc} (search: {search_model}, matching: {matching_model})...")
        start_time = asyncio.get_event_loop().time()
        try:
            result = await find_anatomic_locations(
                finding_name="myocardial infarction",
                description="Death of heart muscle tissue due to lack of blood supply",
                search_model=search_model,
                matching_model=matching_model,
            )
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            print(f"    ✅ Primary: {result.primary_location.concept_text}")
            print(f"    ⏱️  Duration: {duration:.2f}s")
        except Exception as e:
            print(f"    ❌ Error: {e}")

    print()


async def main() -> None:
    """Main entry point for manual testing."""

    import sys

    print("🚀 Starting anatomic location search manual test")
    print("=" * 60)
    print()

    # Check if we have the required environment
    try:
        from findingmodel.config import settings

        if not settings.openai_api_key:
            print("❌ OPENAI_API_KEY is required but not set")
            return
        if not settings.lancedb_uri:
            print("❌ LanceDB configuration is required")
            return
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return

    TEST_MAP = {
        "pcl": test_pcl_tear,
        "heart": test_heart_chamber_enlargement,
        "ptx": test_pneumothorax,
        "brain": test_ependymoma,
        "edge": test_edge_cases,
        "perf": test_performance,
    }

    test_routine = test_pcl_tear
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in TEST_MAP:
            test_routine = TEST_MAP[arg]
        else:
            print(f"❌ Unknown test case '{arg}'. Valid options: {list(TEST_MAP.keys())}")
            return

    # Run the selected test scenario
    await test_routine()

    print("✨ Manual testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
