import dotenv
import logfire

from findingmodel import logger
from findingmodel.index import Index
from findingmodel.tools import describe_finding_name, find_similar_models
from findingmodel.tools.similar_finding_models import SimilarModelAnalysis


async def main() -> SimilarModelAnalysis:
    finding_info = await describe_finding_name(finding_name, model_name="gpt-4o-mini")

    # Find similar models
    index = Index()
    assert await index.count() > 1900, "Index should have more than 1900 entries"
    similar_models = await find_similar_models(
        finding_name, finding_info.description, finding_info.synonyms, index=index
    )
    return similar_models


if __name__ == "__main__":
    import asyncio
    import sys

    if len(sys.argv) < 2 or len(sys.argv[1]) < 5:
        print("Usage: uv run test_find_similar.py <finding_name>")
        sys.exit(1)
    finding_name = sys.argv[1]

    envfile = dotenv.dotenv_values()
    logger.enable("findingmodel")
    logfire.configure(token=envfile["LOGFIRE_TOKEN"])
    logfire.instrument_pydantic_ai()
    logfire.instrument_openai()
    logfire.instrument_pymongo()
    logger.configure(handlers=[logfire.loguru_handler()])

    analysis = asyncio.run(main())

    if len(analysis.similar_models) == 0:
        print(f"No similar models found for finding '{finding_name}'")
    else:
        print(f"Found {len(analysis.similar_models)} similar models for finding '{finding_name}':")
        for model in analysis.similar_models:
            print(f"- {model['oifm_id']} : {model['name']}")
        print(f"Recommendation: {analysis.recommendation} (confidence: {analysis.confidence})")
