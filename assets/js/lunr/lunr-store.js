---
layout: null
---

var store = [
  {%- for page in site.pages -%}
    {%- if forloop.last -%}
      {%- assign l = true -%}
    {%- endif -%}
    {%- unless page.exclude -%}
    {%- if page.title -%}

      {%- if page.header.teaser -%}
        {%- capture teaser -%}{{ page.header.teaser }}{%- endcapture -%}
      {%- else -%}
        {%- assign teaser = site.teaser -%}
      {%- endif -%}

      {
        "title": {{ page.title | jsonify }},
        "excerpt":
          {%- if site.search_full_content == true -%}
            {{ page.content | newline_to_br |
              replace:"<br />", " " |
              replace:"</p>", " " |
              replace:"</h1>", " " |
              replace:"</h2>", " " |
              replace:"</h3>", " " |
              replace:"</h4>", " " |
              replace:"</h5>", " " |
              replace:"</h6>", " "|
              replace:"`", " "|
              replace:"```", " "|
              replace:"$", " "|
              replace:"#", " "|
              replace:"**", " "|
              replace:"__", " "|
              replace:"[", " "|
              replace:"]", " "|
              replace:".", " "|
              replace:"/", " "|
            strip_html | strip_newlines | jsonify }},
          {%- else -%}
            {{ page.content | newline_to_br |
              replace:"<br />", " " |
              replace:"</p>", " " |
              replace:"</h1>", " " |
              replace:"</h2>", " " |
              replace:"</h3>", " " |
              replace:"</h4>", " " |
              replace:"</h5>", " " |
              replace:"</h6>", " "|
              replace:"`", " "|
              replace:"```", " "|
              replace:"$", " "|
              replace:"#", " "|
              replace:"**", " "|
              replace:"__", " "|
              replace:"[", " "|
              replace:"]", " "|
              replace:".", " "|
              replace:"/", " "|
            strip_html | strip_newlines | truncatewords: 50 | jsonify }},
          {%- endif -%}
        "categories": {{ page.categories | jsonify }},
        "tags": {{ page.tags | jsonify }},
        "url": {{ page.url | absolute_url | jsonify }},
        "teaser": {{ teaser | absolute_url | jsonify }}
      }

      {%- unless forloop.last and l -%},{%- endunless -%}

    {%- endif -%}
    {%- endunless -%}
  {%- endfor -%}]
