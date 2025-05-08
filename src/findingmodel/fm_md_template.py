from jinja2 import Template

UNIFIED_MARKDOWN_TEMPLATE_TEXT = """
# {{ name | capitalize }}{% if show_ids and oifm_id %}—`{{ oifm_id }}`{% endif %}

{% if synonyms %}

**Synonyms:** {{ synonyms | join(", ") }}
{% endif %}
{% if tags %}

**Tags:** {{ tags | join(", ") }}
{% endif %}
{% if description %}

{{ description }}
{% endif %}
{% if index_codes_str %}

**Codes:** {{ index_codes_str }}
{% endif %}

## Attributes

{% for attribute in attributes %}
### {{ attribute.name | capitalize }}
{%- if show_ids and attribute.oifma_id is defined -%}—`{{ attribute.oifma_id }}`{%+ endif +%}

{% if attribute.description %}{{ attribute.description }}  {%+ endif -%}
{% if attribute.index_codes_str +%}
**Codes**: {{ attribute.index_codes_str }}  
{% else %}

{%+ endif -%}
{%- if attribute.type == "choice" -%}
{%- if attribute.max_selected and attribute.max_selected > 1 -%}
*(Select up to {{ attribute.max_selected }})*
{%- else %}
*(Select one)*
{% endif %}

{% for value in attribute.values %}
- **{{ value.name }}**{% if value.description %}: {{ value.description }}{%+ endif +%}  
{% if value.index_codes_str %}
_{{ value.index_codes_str }}_
{% endif %}
{% endfor %}
{% elif attribute.type == "numeric" %}
{% if attribute.minimum is defined %}
Mininum: {{ attribute.minimum }}  
{% endif %}
{% if attribute.maximum %}
Maximum: {{ attribute.maximum }}  
{% endif %}
{% if attribute.unit %}
Unit: {{ attribute.unit }}
{% endif %}
{% endif %}

{% endfor %}{%- if footer -%}{{ footer }}{% endif -%}
"""

UNIFIED_MARKDOWN_TEMPLATE = Template(UNIFIED_MARKDOWN_TEMPLATE_TEXT, trim_blocks=True, lstrip_blocks=True)
