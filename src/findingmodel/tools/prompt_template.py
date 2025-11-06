import re
from pathlib import Path
from typing import Any

from jinja2 import Template

PROMPT_TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"


def load_prompt_template(template_file_name: str) -> Template:
    template_file_name = (
        template_file_name if template_file_name.endswith(".md.jinja") else f"{template_file_name}.md.jinja"
    )
    template_file = PROMPT_TEMPLATE_DIR / template_file_name
    if not template_file.exists():
        raise FileNotFoundError(f"Prompt template {template_file_name} not found")
    template_text = template_file.read_text()
    return Template(template_text)


def render_agent_prompt(template: Template, **kwargs: Any) -> tuple[str, str]:  # noqa: ANN401
    """Render prompt template for Pydantic AI Agent pattern.

    Extracts SYSTEM section as instructions and USER section as user prompt.
    This is the preferred method for Pydantic AI agents.

    Args:
        template: Jinja2 template to render
        **kwargs: Variables to pass to template rendering

    Returns:
        Tuple of (instructions, user_prompt)

    Raises:
        ValueError: If template doesn't include USER section

    Example:
        >>> template = load_prompt_template("my_template")
        >>> instructions, user_prompt = render_agent_prompt(template, var1="value")
        >>> agent = Agent(model=..., instructions=instructions, ...)
        >>> result = await agent.run(user_prompt)
    """
    rendered = template.render(**kwargs)

    # Split on markdown headers: # SYSTEM, # USER, # ASSISTANT
    sections = re.split(r"(?:^|\n)# (SYSTEM|USER|ASSISTANT)", rendered)

    instructions = ""
    user_prompt = ""

    # sections[0] is text before first header (usually empty)
    # sections[1::2] are role names (SYSTEM, USER, etc.)
    # sections[2::2] are content for each role
    for i in range(1, len(sections), 2):
        role = sections[i]
        content = sections[i + 1].strip() if i + 1 < len(sections) else ""

        if role == "SYSTEM":
            instructions = content
        elif role == "USER":
            user_prompt = content

    if not user_prompt:
        raise ValueError("Prompt template must include a USER section")

    return instructions, user_prompt
