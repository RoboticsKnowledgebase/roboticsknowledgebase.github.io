---
title: Wiki Index
exclude: true
---
{% include google-search.html %}

<div class="row">
{% for page in site.pages %}
{% unless page.exclude %}
{% if page.title %}
<div>
<h2><a href="{{page.url}}">{{ page.title }}</a></h2>
{% assign excerpt = page.content | split: site.excerpt_separator %}
<p>{{ excerpt[0] | truncatewords:50 | strip_html }}</p>
</div>
{% endif %}
{% endunless %}
{% endfor %}
</div>
